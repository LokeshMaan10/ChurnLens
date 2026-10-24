"""Cox proportional-hazards model for time to churn on subscription spells.

Spells come from Phase 3's builder (pre-cutoff transactions; 30-day gap = churn).
Covariates describe the start of each spell, so fitting (historic spells) and scoring
(each user's current spell) use the same definitions. Hazard ratios are associations
in observational data, not causal effects.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from churnlens.analysis.survival import build_spells

DURATION, EVENT = "duration", "event"


def covariates(spells: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Spell-start covariates. Reference levels: manual renew, full price, 30-day plan,
    any channel outside `channels`."""
    days = spells["first_plan_days"].astype(float)
    paid = spells["first_paid"].astype(float)
    out = pd.DataFrame({
        "auto_renew": (spells["first_auto_renew"] == 1).astype(int),
        "free_start": ((paid == 0) & (spells["first_list_price"] > 0)).astype(int),
        "list_price_zero": (spells["first_list_price"] == 0).astype(int),
        "discounted_start": ((paid > 0) & (paid < spells["first_list_price"])).astype(int),
        "plan_under_30d": (days < 30).astype(int),
        "plan_over_30d": (days > 30).astype(int),
        "price_per_day": np.where(days > 0, paid / days.where(days > 0, 1), 0.0),
    }, index=spells.index)
    for ch in channels:
        out[f"channel_{ch.removeprefix('channel ')}"] = (spells["channel"] == ch).astype(int)
    return out


def top_channels(spells: pd.DataFrame, n: int) -> list[str]:
    """Largest registration channels (excluding missing profiles)."""
    c = spells.loc[spells["channel"] != "no members row", "channel"].value_counts()
    return list(c.index[:n])


def fit(con: duckdb.DuckDBPyConnection, cutoff: dt.date, cfg: dict[str, Any]
        ) -> tuple[CoxPHFitter, list[str], pd.DataFrame]:
    """Fit the Cox model on all sampled users' spells before `cutoff`."""
    spells = build_spells(con, cutoff, cfg, drop_left_truncated=True)
    channels = top_channels(spells, cfg["top_channels"])
    df = covariates(spells, channels).assign(**{DURATION: spells["duration"].astype(float),
                                                EVENT: spells["event"].astype(int)})
    # lifelines needs strictly positive durations; same-day spells get a tiny duration.
    df[DURATION] = df[DURATION].clip(lower=0.5)
    model = CoxPHFitter(penalizer=cfg["penalizer"]).fit(df, DURATION, EVENT)
    return model, channels, df


def ph_test(model: CoxPHFitter, df: pd.DataFrame, sample: int, seed: int) -> pd.DataFrame:
    """Schoenfeld-residual test of proportional hazards per covariate.

    lifelines computes residuals on the data a model was fitted to, so the same
    specification is refitted on a seeded sample and tested there.
    """
    s = df.sample(n=min(sample, len(df)), random_state=seed)
    refit = CoxPHFitter(penalizer=model.penalizer).fit(s, DURATION, EVENT)
    res = proportional_hazard_test(refit, s, time_transform="rank")
    return res.summary[["test_statistic", "p"]].rename_axis("covariate").reset_index()


def current_spells(con: duckdb.DuckDBPyConnection, cutoff: dt.date, users: pd.Series,
                   cfg: dict[str, Any]) -> pd.DataFrame:
    """Each user's latest spell at `cutoff` (left-truncated spells kept: age is a lower
    bound). Users with no pre-cutoff transactions are absent."""
    spells = build_spells(con, cutoff, cfg, drop_left_truncated=False)
    spells = spells[spells["msno"].isin(set(users))]
    latest = spells.sort_values(["msno", "spell_id"]).groupby("msno").tail(1)
    return latest.set_index("msno")


def conditional_survival(model: CoxPHFitter, X: pd.DataFrame, age_days: np.ndarray,
                         months: int, days_per_month: int) -> np.ndarray:
    """s[i, j] = P(still subscribed j months from now | subscribed now), j = 0..months-1,
    i.e. S(age + j*30) / S(age) from the Cox model."""
    times = [j * days_per_month for j in range(months)]
    sf = model.predict_survival_function(X, times=times, conditional_after=age_days)
    s = sf.to_numpy().T
    s[:, 0] = 1.0
    return s
