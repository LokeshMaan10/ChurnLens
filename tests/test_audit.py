"""Audit checks on tiny hand-built tables with known answers."""

import datetime as dt

import duckdb
import pytest

from churnlens.analysis.audit import build_checks, cross_table_checks, profile_table, run_checks

AUDIT_CFG = {
    "plausible_age": {"min": 10, "max": 90},
    "max_seconds_per_day": 86400,
    "min_plausible_date": dt.date(2004, 1, 1),
}
CUTOFFS = {"A": 20170201, "B": 20170301}
TXN_COLS = ("msno VARCHAR, payment_method_id INT, payment_plan_days INT, plan_list_price INT, "
            "actual_amount_paid INT, is_auto_renew INT, transaction_date INT, "
            "membership_expire_date INT, is_cancel INT")


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("CREATE TABLE members_v3 (msno VARCHAR, city INT, bd INT, gender VARCHAR, "
              "registered_via INT, registration_init_time INT)")
    c.execute("""INSERT INTO members_v3 VALUES
        ('a', 1, 0, '', 7, 20150101), ('b', 1, 25, 'male', 7, 20170215),
        ('c', 1, 5, 'female', 7, 20160101), ('d', 1, 120, '', 7, 20170305)""")
    c.execute(f"CREATE TABLE transactions ({TXN_COLS})")
    c.execute("""INSERT INTO transactions VALUES
        ('a', 41, 30, 149, 149, 1, 20170105, 20170205, 0),
        ('a', 41, 30, 149, 149, 1, 20170105, 20170205, 0),
        ('b', 41, 0, 0, 0, 0, 20170210, 19700101, 1),
        ('c', 41, 30, 99, 149, 0, 20170120, 20170110, 0)""")
    c.execute(f"CREATE TABLE transactions_v2 ({TXN_COLS})")
    c.execute("""INSERT INTO transactions_v2 VALUES
        ('a', 41, 30, 149, 149, 1, 20170105, 20170305, 0),
        ('d', 41, 30, 149, 149, 1, 20170310, 20170410, 0)""")
    c.execute("CREATE TABLE user_logs (msno VARCHAR, date INT, num_25 INT, num_50 INT, num_75 INT, "
              "num_985 INT, num_100 INT, num_unq INT, total_secs DOUBLE)")
    c.execute("""INSERT INTO user_logs VALUES
        ('a', 20170101, 1, 0, 0, 0, 2, 3, 500.0), ('a', 20170101, 0, 0, 0, 0, 1, 1, 200.0),
        ('b', 20170102, 0, 0, 0, 0, 1, 5, 99999.0)""")
    return c


def counts(con: duckdb.DuckDBPyConnection, tables: list[str]) -> dict[str, int]:
    rows = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    results = run_checks(con, build_checks(tables, AUDIT_CFG, CUTOFFS), rows)
    return {f"{r.table}.{r.name}": r.count for r in results}


def test_member_checks(con: duckdb.DuckDBPyConnection) -> None:
    got = counts(con, ["members_v3"])
    assert got["members_v3.bd_zero_or_negative"] == 1
    assert got["members_v3.bd_below_min"] == 1
    assert got["members_v3.bd_above_max"] == 1
    assert got["members_v3.gender_missing"] == 2
    assert got["members_v3.registered_on_or_after_cutoff_A"] == 2
    assert got["members_v3.registered_on_or_after_cutoff_B"] == 1


def test_transaction_checks(con: duckdb.DuckDBPyConnection) -> None:
    got = counts(con, ["transactions"])
    assert got["transactions.list_price_zero"] == 1
    assert got["transactions.plan_days_zero"] == 1
    assert got["transactions.paid_above_list"] == 1
    assert got["transactions.expire_before_transaction"] == 2  # 1970 expiry + c's early expiry
    assert got["transactions.expire_implausible_date"] == 1
    assert got["transactions.transaction_on_or_after_cutoff_A"] == 1


def test_log_checks_and_key_duplicates(con: duckdb.DuckDBPyConnection) -> None:
    got = counts(con, ["user_logs"])
    assert got["user_logs.total_secs_above_day"] == 1
    assert got["user_logs.unique_exceeds_plays"] == 1  # 5 unique songs from 1 play
    profile = profile_table(con, "user_logs", ["msno", "date"], ["date"])
    assert profile["key_duplicates"] == 1
    assert profile["full_row_duplicates"] == 0


def test_full_row_duplicates_and_v1_v2_restatement(con: duckdb.DuckDBPyConnection) -> None:
    profile = profile_table(con, "transactions", [], ["transaction_date"])
    assert profile["full_row_duplicates"] == 1
    tables = ["transactions", "transactions_v2"]
    rows = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    cross = {r.name: r.count for r in cross_table_checks(con, tables, rows, CUTOFFS)}
    assert cross["identical_rows_in_v1"] == 0
    assert cross["same_msno_date_rows_differ"] == 1  # same txn, later expiry in v2
    assert cross["rows_before_cutoff_B"] == 1
