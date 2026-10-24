"""Phase 3 helpers: statistics, grouping, spells, calendar overlap."""

import datetime as dt

import duckdb
import pytest

from churnlens.analysis.eda import overlapping_events
from churnlens.analysis.segments import Dimension, band_case, fold_groups
from churnlens.analysis.stats import chi2_independence, two_proportion_z, wilson
from churnlens.analysis.survival import build_spells

SPELL_CFG = {"gap_days": 30, "min_first_spell_start": "2015-03-01"}


def test_wilson_matches_reference_values() -> None:
    r = wilson(10, 100, 0.05)  # textbook Wilson 95% interval for 10/100: 0.0552–0.1744
    assert r.rate == pytest.approx(0.10)
    assert r.lo == pytest.approx(0.0552, abs=1e-4) and r.hi == pytest.approx(0.1744, abs=1e-4)
    assert wilson(0, 50, 0.05).lo == 0.0


def test_chi2_and_two_proportion_agree_for_two_groups() -> None:
    chi2, p_chi, v = chi2_independence([(30, 100), (10, 100)])
    z, p_z = two_proportion_z(30, 100, 10, 100)
    assert chi2 == pytest.approx(z**2) and p_chi == pytest.approx(p_z)
    assert 0 < v < 1


def test_fold_groups_keeps_largest_and_folds_rest() -> None:
    dim = Dimension("x", "X", "x")
    rows = [("a", 5, 1000), ("b", 1, 50), ("c", 3, 500), ("d", 2, 300)]
    out = fold_groups(rows, dim, min_n=200, max_cats=2)
    assert out == [("a", 5, 1000), ("c", 3, 500), ("Other", 3, 350)]


def test_band_case_labels() -> None:
    con = duckdb.connect()
    sql = band_case("v", [10, 20, 30], "", "unknown")
    got = [con.execute(f"SELECT {sql} FROM (SELECT {v} AS v)").fetchone()[0]
           for v in ("15", "20", "NULL", "40")]
    assert got == ["00|10–20", "01|20–30", "99|unknown", "98|out of range"]


@pytest.fixture()
def spell_con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("CREATE TABLE members (msno VARCHAR, registered_via INT)")
    c.execute("CREATE TABLE transactions (msno VARCHAR, payment_method_id INT, "
              "payment_plan_days INT, plan_list_price INT, actual_amount_paid INT, "
              "is_auto_renew INT, transaction_date DATE, membership_expire_date DATE, "
              "is_cancel INT)")
    c.execute("""INSERT INTO transactions VALUES
        -- u1: two same-day rows differing only in amount (the tie that once split spells)
        ('u1', 41, 30, 99, 0,  1, DATE '2016-04-22', DATE '2016-05-22', 0),
        ('u1', 41, 30, 99, 99, 1, DATE '2016-04-22', DATE '2016-05-22', 0),
        ('u1', 41, 30, 149, 149, 1, DATE '2016-05-22', DATE '2016-06-22', 0),
        -- u2: spell ends 2016-06-30, next starts 2016-09-01 (> 30-day gap) -> 2 spells
        ('u2', 41, 30, 149, 149, 0, DATE '2016-05-31', DATE '2016-06-30', 0),
        ('u2', 41, 30, 149, 149, 0, DATE '2016-09-01', DATE '2016-10-01', 0),
        -- u3: first seen before min start -> excluded (may continue an unseen spell)
        ('u3', 41, 30, 149, 149, 0, DATE '2015-01-05', DATE '2015-02-05', 0)""")
    return c


def test_spells_split_on_gaps_and_ties_do_not_split(spell_con) -> None:
    cutoff = dt.date(2017, 2, 1)
    for _ in range(20):  # the old bug was intermittent
        s = build_spells(spell_con, cutoff, SPELL_CFG).set_index(["msno", "spell_id"])
        assert len(s.loc["u1"]) == 1
        assert len(s.loc["u2"]) == 2
        assert "u3" not in s.index.get_level_values(0)
    first_u2 = s.loc[("u2", 1)]
    assert first_u2["event"] and first_u2["duration"] == 30
    last_u2 = s.loc[("u2", 2)]
    assert last_u2["event"]  # 2016-10-01 + 30 days is before the cutoff


def test_spell_open_at_cutoff_is_censored(spell_con) -> None:
    s = build_spells(spell_con, dt.date(2016, 6, 1), SPELL_CFG).set_index(["msno", "spell_id"])
    assert not s.loc[("u1", 1)]["event"]


def test_calendar_overlap() -> None:
    events = [{"name": "LNY", "start": dt.date(2017, 1, 27), "end": dt.date(2017, 2, 1)}]
    assert overlapping_events("2017-02-01", 7, events) == ["LNY"]
    assert overlapping_events("2017-03-01", 7, events) == []
