"""Subscription-adapted RFM: raw values, tie-aware quantile scores, rule-based segments.

Raw values come only from a snapshot's directory (data before its cutoff).
Score cut-points are fitted on the reference snapshot (A) and reused unchanged for
other snapshots, so a test snapshot is never scored against its own distribution.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

CEIL_TOLERANCE = 1e-9
SCORED = {  # raw column -> (score column, orientation: +1 higher is better, -1 lower is better)
    "recency_days": ("r_score", -1),
    "frequency_days": ("f_score", 1),
    "monetary": ("m_score", 1),
}


def raw_rfm(con: duckdb.DuckDBPyConnection, snap_dir: Path, cutoff: dt.date,
            cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per snapshot user: recency, frequency, monetary, personal recency."""
    s, c = snap_dir.as_posix(), f"DATE '{cutoff.isoformat()}'"
    cap, fw, mw = (int(cfg["recency_cap_days"]), int(cfg["frequency_window_days"]),
                   int(cfg["monetary_window_days"]))
    min_days = int(cfg["personal_recency"]["min_active_days"])
    window = max(cap, fw)
    return con.execute(f"""
        WITH lab AS (SELECT msno FROM read_parquet('{s}/labels.parquet')),
        logs AS (
            SELECT msno, date FROM read_parquet('{s}/user_logs/**/*.parquet',
                                                hive_partitioning = true)
            WHERE date >= {c} - INTERVAL {window} DAY
        ),
        last AS (SELECT msno, MAX(date) AS last_date FROM logs
                 WHERE date >= {c} - INTERVAL {cap} DAY GROUP BY 1),
        act AS (SELECT msno, date FROM logs WHERE date >= {c} - INTERVAL {fw} DAY),
        gaps AS (SELECT msno, date_diff('day', LAG(date) OVER (PARTITION BY msno ORDER BY date),
                                        date) AS gap FROM act),
        routine AS (SELECT msno, MEDIAN(gap) AS median_gap, COUNT(*) + 1 AS n_days
                    FROM gaps WHERE gap IS NOT NULL GROUP BY 1),
        freq AS (SELECT msno, COUNT(*) AS frequency_days FROM act GROUP BY 1),
        pay AS (
            SELECT msno, SUM(actual_amount_paid) AS monetary
            FROM read_parquet('{s}/transactions.parquet')
            WHERE is_cancel = 0 AND transaction_date >= {c} - INTERVAL {mw} DAY GROUP BY 1
        )
        SELECT l.msno,
               COALESCE(date_diff('day', last.last_date, {c}), {cap + 1}) AS recency_days,
               COALESCE(freq.frequency_days, 0) AS frequency_days,
               COALESCE(pay.monetary, 0) AS monetary,
               CASE WHEN routine.n_days >= {min_days}
                    THEN date_diff('day', last.last_date, {c}) / routine.median_gap
               END AS personal_recency,
               routine.median_gap
        FROM lab l LEFT JOIN last USING (msno) LEFT JOIN freq USING (msno)
        LEFT JOIN pay USING (msno) LEFT JOIN routine USING (msno)
        ORDER BY l.msno""").df()


@dataclass
class ScoreReference:
    """Empirical distribution of one oriented RFM value in the reference snapshot."""

    values: np.ndarray       # sorted unique oriented values
    cum_share: np.ndarray    # share of reference users at or below each value

    @classmethod
    def fit(cls, oriented: np.ndarray) -> ScoreReference:
        """Build from the reference snapshot's oriented values."""
        values, counts = np.unique(oriented, return_counts=True)
        return cls(values, np.cumsum(counts) / counts.sum())

    def share_at_or_below(self, oriented: np.ndarray) -> np.ndarray:
        """Right-continuous ECDF of the reference evaluated at new values."""
        idx = np.searchsorted(self.values, oriented, side="right") - 1
        return np.where(idx >= 0, self.cum_share[np.clip(idx, 0, None)], 0.0)

    def to_json(self) -> dict[str, list[float]]:
        return {"values": self.values.tolist(), "cum_share": self.cum_share.tolist()}

    @classmethod
    def from_json(cls, d: dict[str, list[float]]) -> ScoreReference:
        return cls(np.array(d["values"]), np.array(d["cum_share"]))


def fit_references(raw: pd.DataFrame) -> dict[str, ScoreReference]:
    """Fit one reference per scored column on the reference snapshot."""
    return {col: ScoreReference.fit(sign * raw[col].to_numpy(dtype=float))
            for col, (_, sign) in SCORED.items()}


def score(raw: pd.DataFrame, refs: dict[str, ScoreReference], n_levels: int) -> pd.DataFrame:
    """Add r/f/m scores: ceil(n_levels * reference share at or below the value), in 1..n."""
    out = raw.copy()
    for col, (score_col, sign) in SCORED.items():
        share = refs[col].share_at_or_below(sign * out[col].to_numpy(dtype=float))
        # Subtract a tiny tolerance so exact boundaries (e.g. 5 * 0.6 = 3.0000000000000004)
        # do not round up into the next score.
        out[score_col] = np.clip(np.ceil(n_levels * share - CEIL_TOLERANCE), 1,
                                 n_levels).astype(int)
    out["rfm_code"] = (out["r_score"].astype(str) + out["f_score"].astype(str)
                       + out["m_score"].astype(str))
    return out


def segment_case_sql(rules: list[dict[str, str]]) -> str:
    """CASE expression implementing first-match-wins segment rules."""
    if not rules or rules[-1]["when"].strip().lower() != "true":
        raise ValueError("segment rules must end with a catch-all rule (when: true)")
    whens = " ".join(f"WHEN {r['when']} THEN '{r['name']}'" for r in rules)
    return f"CASE {whens} END"


def assign_segments(scored: pd.DataFrame, rules: list[dict[str, str]]) -> pd.Series:
    """Apply segment rules (DuckDB SQL predicates) in order; first match wins."""
    con = duckdb.connect()
    con.register("scored", scored)
    seg = con.execute(f"SELECT {segment_case_sql(rules)} AS segment FROM scored").df()["segment"]
    con.close()
    return seg.set_axis(scored.index)


def score_distribution(scored: pd.DataFrame, n_levels: int) -> pd.DataFrame:
    """Share of users at each score level, per dimension (shows unused levels)."""
    rows = []
    for _, (score_col, _) in SCORED.items():
        counts = scored[score_col].value_counts(normalize=True)
        rows.append({"score": score_col, **{str(k): float(counts.get(k, 0.0))
                                            for k in range(1, n_levels + 1)}})
    return pd.DataFrame(rows)


def personal_recency_band(value: float) -> str:
    """Band label for personalised recency (gap relative to the user's own routine)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "too little history"
    if value <= 1:
        return "≤1x usual gap"
    if value <= 3:
        return "1–3x usual gap"
    if value <= 10:
        return "3–10x usual gap"
    return ">10x usual gap"
