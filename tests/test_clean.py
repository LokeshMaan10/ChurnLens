"""Cleaning rules on tiny hand-built raw tables with known answers."""

import datetime as dt

import duckdb
import pandas as pd
import pandera.errors
import pytest

from churnlens.data.clean import (
    CLEAN_TABLES,
    clean_table_sql,
    members_sql,
    transactions_sql,
)
from churnlens.data.schemas import clean_schemas, raw_schemas

RULES = {
    "plausible_age": {"min": 10, "max": 90},
    "max_seconds_per_day": 86400,
    "min_plausible_date": dt.date(2004, 1, 1),
    "transactions_source": "combined",
    "restatement_keep": "v2",
    "v1_coverage_end": dt.date(2017, 2, 28),
}
TXN_COLS = ("msno VARCHAR, payment_method_id INT, payment_plan_days INT, plan_list_price INT, "
            "actual_amount_paid INT, is_auto_renew TINYINT, transaction_date INT, "
            "membership_expire_date INT, is_cancel TINYINT")


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"CREATE TABLE raw_transactions ({TXN_COLS})")
    c.execute("""INSERT INTO raw_transactions VALUES
        ('a', 41, 30, 149, 149, 1, 20170105, 20170205, 0),
        ('a', 41, 30, 149, 149, 1, 20170105, 20170205, 0),   -- exact duplicate within v1
        ('b', 41, 30, 149, 149, 0, 20170110, 20170210, 0),   -- restated in v2
        ('c', 41, 30, 149, 149, 0, 20170112, 19700101, 0),   -- implausible expiry
        ('e', 41, 30, 149, 149, 0, 20170115, 20170215, 0)""")  # identical in v2
    c.execute(f"CREATE TABLE raw_transactions_v2 ({TXN_COLS})")
    c.execute("""INSERT INTO raw_transactions_v2 VALUES
        ('b', 41, 30, 149, 149, 0, 20170110, 20170410, 0),
        ('e', 41, 30, 149, 149, 0, 20170115, 20170215, 0),
        ('d', 41, 30, 149, 149, 1, 20170310, 20170410, 0)""")
    c.execute("CREATE TABLE raw_members_v3 (msno VARCHAR, city INT, bd INT, gender VARCHAR, "
              "registered_via INT, registration_init_time INT)")
    c.execute("INSERT INTO raw_members_v3 VALUES ('a', 1, 0, '', 7, 20150101), "
              "('b', 1, 30, 'male', 7, 20160101), ('c', 1, 150, 'female', 7, 20160102)")
    for view in ("raw_user_logs", "raw_user_logs_v2"):
        c.execute(f"CREATE TABLE {view} (msno VARCHAR, date INT, num_25 INT, num_50 INT, "
                  "num_75 INT, num_985 INT, num_100 INT, num_unq INT, total_secs DOUBLE)")
    c.execute("INSERT INTO raw_user_logs VALUES ('a', 20170101, 1, 0, 0, 0, 2, 3, 500.0), "
              "('b', 20170102, 0, 0, 0, 0, 1, 1, 99999.0)")
    c.execute("INSERT INTO raw_user_logs_v2 VALUES ('z', 20170305, 0, 0, 0, 0, 4, 4, 900.0)")
    return c


def txns(con: duckdb.DuckDBPyConnection, **overrides: str) -> pd.DataFrame:
    return con.execute(transactions_sql({**RULES, **overrides})).df().sort_values("msno")


def test_combined_dedup_and_restatement_keeps_v2(con: duckdb.DuckDBPyConnection) -> None:
    out = txns(con)
    assert list(out["msno"]) == ["a", "b", "c", "d", "e"]  # duplicate 'a' collapsed
    b = out[out.msno == "b"].iloc[0]
    assert b.restated and b.source == "v2"
    assert b.membership_expire_date == pd.Timestamp("2017-04-10")
    assert out[out.msno == "e"].iloc[0].source == "both"
    c = out[out.msno == "c"].iloc[0]
    assert c.expire_implausible and pd.isna(c.membership_expire_date)


def test_restatement_keep_v1_and_v1_only(con: duckdb.DuckDBPyConnection) -> None:
    b = txns(con, restatement_keep="v1").query("msno == 'b'").iloc[0]
    assert b.restated and b.source == "v1"
    assert b.membership_expire_date == pd.Timestamp("2017-02-10")
    v1 = txns(con, transactions_source="v1_only")
    assert "d" not in set(v1.msno) and set(v1.source) == {"v1"}
    assert not v1.restated.any()


def test_v1_then_v2_takes_history_from_v1_only(con: duckdb.DuckDBPyConnection) -> None:
    out = txns(con, transactions_source="v1_then_v2").set_index("msno")
    # 'b' is restated in v2 with a later expiry: the v1 version must win, unflagged.
    assert out.loc["b", "membership_expire_date"] == pd.Timestamp("2017-02-10")
    assert out.loc["b", "source"] == "v1" and not out["restated"].any()
    # 'e' sits identically in both files but inside v1's period: it counts as v1 only.
    assert out.loc["e", "source"] == "v1"
    # 'd' is dated after v1's coverage, so it comes from v2.
    assert out.loc["d", "source"] == "v2"
    inside_v1 = out[out["transaction_date"] <= pd.Timestamp("2017-02-28")]
    assert set(inside_v1["source"]) == {"v1"}


def test_members_age_and_gender(con: duckdb.DuckDBPyConnection) -> None:
    out = con.execute(members_sql(RULES)).df().set_index("msno")
    assert pd.isna(out.loc["a", "age"]) and pd.isna(out.loc["a", "gender"])
    assert out.loc["b", "age"] == 30 and not out.loc["b", "age_invalid"]
    assert pd.isna(out.loc["c", "age"]) and out.loc["c", "age_invalid"]


def test_logs_cap_and_scoring_only_is_separate(con: duckdb.DuckDBPyConnection) -> None:
    logs = con.execute(clean_table_sql("user_logs", RULES)).df().set_index("msno")
    assert logs.loc["b", "total_secs"] == 86400 and logs.loc["b", "secs_capped"]
    assert "z" not in logs.index, "March scoring logs must not enter the training log table"
    assert CLEAN_TABLES["user_logs"][0] == "user_logs"
    assert CLEAN_TABLES["user_logs_scoring"][2].startswith("scoring_only/")
    scoring = con.execute(clean_table_sql("user_logs_scoring", RULES)).df()
    assert list(scoring.msno) == ["z"]


def test_clean_schema_accepts_clean_output(con: duckdb.DuckDBPyConnection) -> None:
    schemas = clean_schemas(RULES)
    schemas["transactions"].validate(txns(con), lazy=True)
    members = con.execute(members_sql(RULES)).df()
    members["age"] = members["age"].astype(float)
    schemas["members"].validate(members, lazy=True)


def test_schemas_reject_bad_rows() -> None:
    raw_labels = raw_schemas()["labels"]
    bad = pd.DataFrame({"msno": ["a"], "is_churn": pd.Series([2], dtype="int8")})
    with pytest.raises(pandera.errors.SchemaErrors):
        raw_labels.validate(bad, lazy=True)
    raw_logs = raw_schemas()["logs"]
    bad_date = pd.DataFrame({
        "msno": ["a"], "date": pd.Series([20170231], dtype="int32"),
        **{c: pd.Series([0], dtype="int32")
           for c in ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq"]},
        "total_secs": [1.0]})
    with pytest.raises(pandera.errors.SchemaErrors):
        raw_logs.validate(bad_date, lazy=True)
