"""Phase 4: RFM raw values, tie-aware scoring, segment rules, K-Means helpers."""

import datetime as dt
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from churnlens.segmentation import kmeans, rfm

CFG = {"recency_cap_days": 90, "frequency_window_days": 90, "monetary_window_days": 180,
       "personal_recency": {"min_active_days": 3}}
CUTOFF = dt.date(2017, 2, 1)


@pytest.fixture()
def snap(tmp_path: Path) -> Path:
    c = duckdb.connect()
    c.execute("COPY (SELECT * FROM (VALUES ('a', 0), ('b', 1), ('c', 0)) t(msno, is_churn)) "
              f"TO '{(tmp_path / 'labels.parquet').as_posix()}' (FORMAT parquet)")
    logs = tmp_path / "user_logs" / "month=2017-01"
    logs.mkdir(parents=True)
    # a: active every 2nd day, last listen 2017-01-31 (recency 1, median gap 2)
    # b: active 3 days long ago (outside the 90-day window) -> dormant
    c.execute(f"""COPY (
        SELECT 'a' AS msno, DATE '2017-01-31' - INTERVAL (2 * i) DAY AS date FROM range(10) r(i)
        UNION ALL SELECT 'b', DATE '2016-10-01' + INTERVAL (i) DAY FROM range(3) r(i)
        ) TO '{(logs / 'part-0.parquet').as_posix()}' (FORMAT parquet)""")
    c.execute(f"""COPY (SELECT * FROM (VALUES
        ('a', DATE '2017-01-01', 149, 0), ('a', DATE '2016-01-01', 149, 0),
        ('a', DATE '2017-01-10', 149, 1), ('c', DATE '2016-12-01', 99, 0)
        ) t(msno, transaction_date, actual_amount_paid, is_cancel))
        TO '{(tmp_path / 'transactions.parquet').as_posix()}' (FORMAT parquet)""")
    return tmp_path


def test_raw_rfm_values(snap: Path) -> None:
    raw = rfm.raw_rfm(duckdb.connect(), snap, CUTOFF, CFG).set_index("msno")
    assert raw.loc["a", "recency_days"] == 1 and raw.loc["a", "frequency_days"] == 10
    assert raw.loc["a", "monetary"] == 149          # old and cancelled payments excluded
    assert raw.loc["a", "personal_recency"] == pytest.approx(0.5)  # 1-day gap / median 2
    assert raw.loc["b", "recency_days"] == 91       # nothing in 90 days -> cap + 1
    assert raw.loc["b", "frequency_days"] == 0 and np.isnan(raw.loc["b", "personal_recency"])
    assert raw.loc["c", "monetary"] == 99


def test_scores_are_tie_aware_and_top_value_scores_5() -> None:
    raw = pd.DataFrame({"recency_days": [1] * 6 + [2, 3, 50, 91],
                        "frequency_days": list(range(10)), "monetary": [0] * 2 + [149] * 8})
    s = rfm.score(raw, rfm.fit_references(raw), 5)
    assert set(s.loc[raw["recency_days"] == 1, "r_score"]) == {5}   # ties share a score
    assert s.loc[raw["recency_days"] == 91, "r_score"].iloc[0] == 1
    assert s["f_score"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert s.loc[raw["monetary"] == 149, "m_score"].unique().tolist() == [5]


def test_other_snapshot_scored_against_reference_not_itself() -> None:
    ref = pd.DataFrame({"recency_days": [1, 2, 3, 4, 5], "frequency_days": [1, 2, 3, 4, 5],
                        "monetary": [1, 2, 3, 4, 5]})
    other = pd.DataFrame({"recency_days": [5, 5], "frequency_days": [100, 0],
                          "monetary": [3, 3]})
    s = rfm.score(other, rfm.fit_references(ref), 5)
    assert s["f_score"].tolist() == [5, 1]   # beyond the reference range clips to 5 / 1
    assert s["r_score"].tolist() == [1, 1]


RULES = [{"name": "Dormant", "when": "frequency_days = 0"},
         {"name": "Top", "when": "r_score = 5"},
         {"name": "Rest", "when": "true"}]


def test_segments_first_match_wins_and_catch_all_required() -> None:
    df = pd.DataFrame({"frequency_days": [0, 5, 5], "r_score": [5, 5, 1]})
    assert rfm.assign_segments(df, RULES).tolist() == ["Dormant", "Top", "Rest"]
    with pytest.raises(ValueError, match="catch-all"):
        rfm.segment_case_sql(RULES[:2])


def test_configured_rules_cover_everyone() -> None:
    from churnlens.config import load_config
    rules = load_config("rfm")["segments"]
    grid = pd.DataFrame([{"frequency_days": f, "recency_days": r, "monetary": 0,
                          "personal_recency": p, "r_score": rs, "f_score": fs, "m_score": ms}
                         for f in (0, 10) for r in (1, 91) for p in (None, 0.5, 5.0)
                         for rs in range(1, 6) for fs in range(1, 6) for ms in range(1, 6)])
    seg = rfm.assign_segments(grid, rules)
    assert seg.notna().all()
    assert set(seg) <= {r["name"] for r in rules}


def test_personal_recency_bands() -> None:
    assert rfm.personal_recency_band(float("nan")) == "too little history"
    assert rfm.personal_recency_band(0.5) == "≤1x usual gap"
    assert rfm.personal_recency_band(12.0) == ">10x usual gap"


KM_CFG = {
    "population": "monetary > 0 AND frequency_days > 0",
    "excluded_groups": [
        {"name": "no pay, no listen", "when": "monetary = 0 AND frequency_days = 0"},
        {"name": "no pay", "when": "monetary = 0"},
        {"name": "no listen", "when": "frequency_days = 0"},
        {"name": "other", "when": "true"},
    ],
    "k_range": [2, 3], "n_init": 3, "silhouette_sample": 500, "seed": 0,
}


def _toy_population(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    big = pd.DataFrame({"recency_days": rng.integers(1, 3, 2 * n),
                        "frequency_days": rng.integers(60, 90, 2 * n),
                        "monetary": rng.choice([745, 894], 2 * n)})
    small = pd.DataFrame({"recency_days": rng.integers(30, 60, n),
                          "frequency_days": rng.integers(1, 10, n),
                          "monetary": rng.choice([99, 149], n)})
    return pd.concat([big, small], ignore_index=True)


def test_only_payers_who_listened_are_clustered_others_get_rule_labels() -> None:
    pop = _toy_population()
    result = kmeans.select_and_fit(pop, KM_CFG)
    others = pd.DataFrame({"recency_days": [91, 1, 91], "frequency_days": [0, 40, 0],
                           "monetary": [0, 0, 594]})
    labels = kmeans.assign(result, pd.concat([pop.head(3), others], ignore_index=True), KM_CFG)
    assert all(lab.startswith("cluster ") for lab in labels[:3])
    assert labels[3:].tolist() == ["no pay, no listen", "no pay", "no listen"]


def test_cluster_numbers_follow_size() -> None:
    pop = _toy_population()
    result = kmeans.select_and_fit(pop, KM_CFG)
    labels = kmeans.assign(result, pop, KM_CFG)
    sizes = labels.value_counts()
    assert sizes.index[0] == "cluster 1"
    assert sizes.is_monotonic_decreasing


def test_kmeans_compare_detects_added_information() -> None:
    rng = np.random.default_rng(0)
    seg = pd.Series(rng.choice(["x", "y"], 2000))
    clusters = pd.Series(rng.integers(0, 2, 2000))
    churn = pd.Series((clusters == 1) & (rng.random(2000) < 0.6)).astype(int)
    out = kmeans.compare(seg, clusters, churn, min_gain=0.02)
    assert out["keep"] and out["cramers_v_clusters"] > out["cramers_v_segments"]
