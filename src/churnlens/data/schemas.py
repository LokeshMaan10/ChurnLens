"""pandera schemas for the raw and clean layers.

Raw schemas are contracts on what the source files look like (types, valid
YYYYMMDD dates, binary flags); they do not encode cleaning decisions. Clean
schemas encode the Phase 1 rules (age range, capped seconds, parsed dates).

pandera validates per-row and per-column rules in batches. Table-wide
uniqueness (e.g. one row per (msno, date)) cannot be checked batch by batch,
so it is asserted in DuckDB instead (see clean.py: `KEY_ASSERTIONS`).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa

from churnlens.config import date_to_int

BINARY = [0, 1]
# Exact stored dtypes (no coercion), so type drift in the source fails validation.
I8, I32 = "int8", "int32"
LOG_COUNTS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq"]


def _valid_yyyymmdd(series: pd.Series) -> pd.Series:
    """True where an integer is a real calendar date in YYYYMMDD form."""
    parsed = pd.to_datetime(series.astype("string"), format="%Y%m%d", errors="coerce")
    return parsed.notna()


VALID_DATE = pa.Check(_valid_yyyymmdd, element_wise=False, name="valid_yyyymmdd")
MSNO = pa.Column(str, nullable=False)


# ---------------------------------------------------------------- raw contracts


def raw_schemas() -> dict[str, pa.DataFrameSchema]:
    """Schemas for the raw Parquet layer, keyed by file kind."""
    txn = pa.DataFrameSchema({
        "msno": MSNO,
        "payment_method_id": pa.Column(I32),
        "payment_plan_days": pa.Column(I32),
        "plan_list_price": pa.Column(I32),
        "actual_amount_paid": pa.Column(I32),
        "is_auto_renew": pa.Column(I8, pa.Check.isin(BINARY)),
        "transaction_date": pa.Column(I32, VALID_DATE),
        "membership_expire_date": pa.Column(I32, VALID_DATE),
        "is_cancel": pa.Column(I8, pa.Check.isin(BINARY)),
    }, strict=True, name="raw_transactions")
    logs = pa.DataFrameSchema({
        "msno": MSNO,
        "date": pa.Column(I32, VALID_DATE),
        **{c: pa.Column(I32) for c in LOG_COUNTS},
        "total_secs": pa.Column(float),
    }, strict=True, name="raw_logs")
    return {
        "labels": pa.DataFrameSchema(
            {"msno": MSNO, "is_churn": pa.Column(I8, pa.Check.isin(BINARY))},
                                     strict=True, name="raw_labels"),
        "members": pa.DataFrameSchema({
            "msno": MSNO,
            "city": pa.Column(I32),
            "bd": pa.Column(I32),
            "gender": pa.Column(str, pa.Check.isin(["male", "female", ""])),
            "registered_via": pa.Column(I32),
            "registration_init_time": pa.Column(I32, VALID_DATE),
        }, strict=True, name="raw_members"),
        "transactions": txn,
        "logs": logs,
    }


# ---------------------------------------------------------------- clean layer


def clean_schemas(rules: dict[str, Any]) -> dict[str, pa.DataFrameSchema]:
    """Schemas for the clean layer, parameterised by configs/cleaning.yaml."""
    age = rules["plausible_age"]
    min_date = pd.Timestamp(str(date_to_int(rules["min_plausible_date"])))
    max_secs = float(rules["max_seconds_per_day"])
    date = pa.Column(pa.DateTime, coerce=True, nullable=False)
    non_negative = pa.Check.greater_than_or_equal_to(0)
    return {
        "labels": pa.DataFrameSchema(
            {"msno": MSNO, "is_churn": pa.Column(I8, pa.Check.isin(BINARY))},
                                     strict=True, name="clean_labels"),
        "members": pa.DataFrameSchema({
            "msno": MSNO,
            "city": pa.Column(I32),
            "bd_raw": pa.Column(I32),
            # Integer column with NULLs arrives in pandas as float.
            "age": pa.Column(float, pa.Check.in_range(age["min"], age["max"]), nullable=True),
            "gender": pa.Column(str, pa.Check.isin(["male", "female"]), nullable=True),
            "registered_via": pa.Column(I32),
            "registration_date": date,
            "age_invalid": pa.Column(bool),
        }, strict=True, name="clean_members"),
        "transactions": pa.DataFrameSchema({
            "msno": MSNO,
            "payment_method_id": pa.Column(I32),
            "payment_plan_days": pa.Column(I32, non_negative),
            "plan_list_price": pa.Column(I32, non_negative),
            "actual_amount_paid": pa.Column(I32, non_negative),
            "is_auto_renew": pa.Column(I8, pa.Check.isin(BINARY)),
            "transaction_date": date,
            "membership_expire_date": pa.Column(
                pa.DateTime, pa.Check.greater_than_or_equal_to(min_date), coerce=True,
                nullable=True),
            "is_cancel": pa.Column(I8, pa.Check.isin(BINARY)),
            "source": pa.Column(str, pa.Check.isin(["v1", "v2", "both"])),
            "restated": pa.Column(bool),
            "expire_implausible": pa.Column(bool),
        }, strict=True, name="clean_transactions"),
        "logs": pa.DataFrameSchema({
            "msno": MSNO,
            "date": date,
            **{c: pa.Column(I32, non_negative) for c in LOG_COUNTS},
            "total_secs": pa.Column(float, pa.Check.in_range(0.0, max_secs)),
            "secs_capped": pa.Column(bool),
        }, strict=True, name="clean_logs"),
    }
