"""Phase 5: feature values on a hand-built snapshot, registry consistency, leakage guards."""

from __future__ import annotations

import datetime as dt
import inspect
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from churnlens.config import load_config
from churnlens.features import behavior, build, members, transactions
from churnlens.features.registry import all_specs

CUTOFF = dt.date(2017, 2, 1)
CFG = load_config("features")
SPECS = all_specs(CFG)
LOG_COLS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq"]


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _txn(msno: str, tdate: str, expiry: str, paid: int = 149, price: int = 149,
         days: int = 30, auto: int = 1, cancel: int = 0, method: int = 41) -> dict:
    return {"msno": msno, "payment_method_id": method, "payment_plan_days": days,
            "plan_list_price": price, "actual_amount_paid": paid, "is_auto_renew": auto,
            "transaction_date": _d(tdate), "membership_expire_date": _d(expiry),
            "is_cancel": cancel, "source": "v1", "restated": False, "expire_implausible": False}


def _log(msno: str, date: str, minutes: float, n25: int = 1, n100: int = 3) -> dict:
    return {"msno": msno, "date": _d(date), "num_25": n25, "num_50": 0, "num_75": 0,
            "num_985": 0, "num_100": n100, "num_unq": 2, "total_secs": minutes * 60.0,
            "secs_capped": False}


def write_snapshot(root: Path, logs: list[dict], txns: list[dict]) -> Path:
    """Minimal snapshot dir + Phase 4 RFM file, with the same schemas as the real ones."""
    snap = root / "A"
    (snap / "user_logs" / "month=2017-01").mkdir(parents=True)
    pd.DataFrame({"msno": ["u1", "u2"], "is_churn": [0, 1]}).to_parquet(
        snap / "labels.parquet", index=False)
    pd.DataFrame(logs).to_parquet(snap / "user_logs" / "month=2017-01" / "part-0.parquet",
                                  index=False)
    pd.DataFrame(txns).to_parquet(snap / "transactions.parquet", index=False)
    pd.DataFrame({
        "msno": ["u1", "u2"], "city": [1, 13], "bd_raw": [30, 0], "age": [30.0, None],
        "gender": ["male", None], "registered_via": [7, 9], "age_invalid": [False, True],
        "registration_date": [_d("2015-02-01"), None],
        "registered_after_cutoff": [False, True]}).to_parquet(snap / "members.parquet",
                                                                index=False)
    pd.DataFrame({"msno": ["u1", "u2"], "r_score": [5, 1], "f_score": [3, 1],
                  "m_score": [3, 1], "personal_recency": [0.5, None],
                  "segment": ["Steady", "Dormant"], "cluster": ["cluster 1", "x"]}).to_parquet(
        root / "rfm_A.parquet", index=False)
    return snap


BASE_LOGS = [_log("u1", "2017-01-31", 60), _log("u1", "2017-01-29", 30),
             _log("u1", "2016-12-01", 10)]
BASE_TXNS = [_txn("u1", "2016-12-31", "2017-01-31"), _txn("u1", "2017-01-15", "2017-02-14"),
             _txn("u1", "2016-06-01", "2016-06-02", cancel=1),
             _txn("u2", "2017-01-10", "2017-01-17", paid=0, price=0, days=7, auto=0)]


@pytest.fixture()
def snapshot(tmp_path: Path) -> Path:
    return write_snapshot(tmp_path, BASE_LOGS, BASE_TXNS)


def built(snap: Path) -> pd.DataFrame:
    return build.build_snapshot(duckdb.connect(), snap, snap.parent / "rfm_A.parquet",
                                CUTOFF, CFG, SPECS).set_index("msno")


def test_builder_output_matches_registry_and_known_values(snapshot: Path) -> None:
    f = built(snapshot)
    assert list(f.columns) == [s.name for s in SPECS]
    u1, u2 = f.loc["u1"], f.loc["u2"]
    assert u1["minutes_7d"] == pytest.approx(90.0) and u1["active_days_7d"] == 2
    assert u1["minutes_90d"] == pytest.approx(100.0)
    assert u1["days_since_last_listen"] == 1
    assert u1["longest_inactive_gap_90d"] == 58      # 2016-12-02 .. 2017-01-28
    assert u1["skip_rate_30d"] == pytest.approx(0.25)  # 2 of 8 plays
    assert u1["last_plan_days"] == 30 and u1["days_to_expiry"] == 13
    assert u1["n_cancels_all"] == 1 and u1["days_since_last_cancel"] == 245
    assert u1["tenure_days"] == (CUTOFF - _d("2015-02-01")).days
    assert u2["days_since_last_listen"] == CFG["recency_cap_days"] + 1
    assert pd.isna(u2["skip_rate_30d"]) and u2["minutes_7d"] == 0
    assert u2["last_is_free"] == 1 and u2["last_list_price_zero"] == 1


def test_slope_and_cv_match_numpy_on_the_dense_daily_series(snapshot: Path) -> None:
    import numpy as np
    u1 = built(snapshot).loc["u1"]
    minutes = np.zeros(91)   # index = days before cutoff (1..90)
    for row in BASE_LOGS:
        minutes[(CUTOFF - row["date"]).days] = row["total_secs"] / 60
    for w in CFG["slope_windows_days"]:
        d = np.arange(1, w + 1)
        expected = np.polyfit(-d, minutes[1:w + 1], 1)[0]
        assert u1[f"minutes_slope_{w}d"] == pytest.approx(expected, abs=1e-7)
    series = minutes[1:91]
    assert u1["minutes_cv_90d"] == pytest.approx(series.std() / series.mean(), abs=1e-7)


def test_post_cutoff_registration_is_indistinguishable_from_missing(snapshot: Path) -> None:
    u2 = built(snapshot).loc["u2"]
    assert u2["member_info_missing"] == 1
    assert pd.isna(u2["tenure_days"]) and pd.isna(u2["city"]) and pd.isna(u2["registered_via"])


def test_rows_on_or_after_cutoff_are_ignored(tmp_path: Path, snapshot: Path) -> None:
    """Defence in depth: even if a snapshot file contained label-window rows, features
    would not change."""
    before = built(snapshot)
    leaky = tmp_path / "leaky"
    shutil.copytree(snapshot.parent, leaky)
    snap = leaky / "A"
    late_logs = BASE_LOGS + [_log("u1", "2017-02-01", 999), _log("u2", "2017-02-20", 999)]
    late_txns = BASE_TXNS + [_txn("u1", "2017-02-01", "2017-12-31", cancel=1),
                             _txn("u2", "2017-02-10", "2018-02-10", paid=1788, days=410)]
    pd.DataFrame(late_logs).to_parquet(snap / "user_logs" / "month=2017-01" / "part-0.parquet",
                                       index=False)
    pd.DataFrame(late_txns).to_parquet(snap / "transactions.parquet", index=False)
    pd.testing.assert_frame_equal(before, built(snap))


def test_feature_sql_only_reads_snapshot_inputs() -> None:
    """Feature SQL must not reference the clean layer or scoring-only data."""
    for module in (behavior, transactions, members):
        source = inspect.getsource(module)
        assert "scoring_only" not in source and "user_logs_scoring" not in source
        assert "churnlens.duckdb" not in source


def test_demographics_are_out_of_the_model_by_default() -> None:
    by_name = {s.name: s for s in SPECS}
    for name in CFG["demographics"]["features"]:
        assert not by_name[name].use_in_model
    assert all(s.use_in_model for s in SPECS if s.group != "demographics")


def test_every_feature_is_documented() -> None:
    for s in SPECS:
        assert s.definition and s.rationale and s.leakage_note
        assert s.leakage_risk in {"low", "medium"} and s.kind in {"numeric", "binary",
                                                                   "categorical"}
