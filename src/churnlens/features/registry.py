"""Feature registry: one documented entry per feature column.

The registry is the single source of truth for feature documentation (definition,
why it matters, leakage risk) and for which features the model may use. The builder
asserts that the columns it produces match this registry exactly.

Leakage risk levels:
  low     derived only from timestamped events strictly before the cutoff
  medium  pre-cutoff events, but a value could have been restated after the cutoff
          (e.g. expiry dates in a later export) or the source is an undated extract
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    """Documentation and usage flags for one feature column."""

    name: str
    group: str
    kind: str              # numeric | binary | categorical
    definition: str
    rationale: str
    leakage_risk: str      # low | medium
    leakage_note: str
    use_in_model: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


LOW = ("low", "Uses only events dated strictly before the cutoff.")
LOGS = ("low", "Daily logs dated strictly before the cutoff; the scoring-only March logs "
               "are never read.")


def listening_specs(cfg: dict[str, Any]) -> list[FeatureSpec]:
    """Listening level, trend, consistency and quality features."""
    windows = cfg["listening_windows_days"]
    longest = max(windows)
    out: list[FeatureSpec] = []
    for w in windows:
        out += [
            FeatureSpec(f"minutes_{w}d", "listening level", "numeric",
                        f"Total minutes listened in the {w} days before the cutoff.",
                        "Overall engagement; lower listening is associated with churn.", *LOGS),
            FeatureSpec(f"active_days_{w}d", "listening level", "numeric",
                        f"Days with any listening in the last {w} days.",
                        "Habit strength, less sensitive to binge days than minutes.", *LOGS),
            FeatureSpec(f"plays_{w}d", "listening level", "numeric",
                        f"Songs played (any completion level) in the last {w} days.",
                        "Engagement measured in plays rather than time.", *LOGS),
            FeatureSpec(f"unique_songs_{w}d", "listening level", "numeric",
                        f"Sum of daily unique songs in the last {w} days.",
                        "Breadth of listening; narrow listening may signal lower attachment.",
                        *LOGS),
        ]
    for p in cfg["ratio_pairs"]:
        r, b = p["recent"], p["base"]
        out += [
            FeatureSpec(f"minutes_ratio_{r}_{b}", "listening trend", "numeric",
                        f"Minutes per day in the last {r} days / minutes per day in the {b} days "
                        "before that. NULL if the earlier period has no listening.",
                        "A drop relative to the user's own recent level (e.g. 0.38 = down 62%).",
                        *LOGS),
            FeatureSpec(f"active_days_ratio_{r}_{b}", "listening trend", "numeric",
                        f"Active-day rate in the last {r} days / rate in the {b} days before.",
                        "Change in habit frequency relative to the user's own baseline.", *LOGS),
        ]
    for w in cfg["slope_windows_days"]:
        out.append(FeatureSpec(
            f"minutes_slope_{w}d", "listening trend", "numeric",
            f"OLS slope of daily minutes (inactive days = 0) over the last {w} days, "
            "in minutes/day per day; positive = rising.",
            "Direction of the listening trajectory, using every day rather than two periods.",
            *LOGS))
    cap = cfg["recency_cap_days"]
    out += [
        FeatureSpec(f"minutes_cv_{longest}d", "engagement consistency", "numeric",
                    f"Coefficient of variation of daily minutes over {longest} days "
                    "(inactive days = 0). NULL if no listening.",
                    "Erratic listening may indicate a weaker routine.", *LOGS),
        FeatureSpec(f"longest_inactive_gap_{longest}d", "engagement consistency", "numeric",
                    f"Longest run of consecutive days without listening in the last {longest} "
                    "days (including a run that reaches the cutoff).",
                    "Long breaks are associated with lapsing.", *LOGS),
        FeatureSpec("days_since_last_listen", "engagement consistency", "numeric",
                    f"Days from the last listening day to the cutoff; {cap + 1} if none in "
                    f"{cap} days.", "Recency of engagement (the R in RFM, unbinned).", *LOGS),
        FeatureSpec("minutes_per_active_day_30d", "listening level", "numeric",
                    "Minutes in the last 30 days / active days in the last 30 days.",
                    "Session depth on days the user does listen.", *LOGS),
        FeatureSpec(f"capped_days_{longest}d", "data quality", "numeric",
                    f"Days in the last {longest} whose total_secs was capped at one day "
                    "(Phase 1 rule L1).", "Flags implausible logging that may distort minutes.",
                    *LOGS),
    ]
    for q in cfg["quality_windows_days"]:
        out += [
            FeatureSpec(f"skip_rate_{q}d", "listening quality", "numeric",
                        f"Share of plays stopped before 25% of the song, last {q} days. NULL "
                        "if no plays.", "Frequent skipping can indicate poor content fit.",
                        *LOGS),
            FeatureSpec(f"completion_rate_{q}d", "listening quality", "numeric",
                        f"Share of plays over 98.5% complete, last {q} days. NULL if no plays.",
                        "Full listens suggest content satisfaction.", *LOGS),
            FeatureSpec(f"unique_song_share_{q}d", "listening quality", "numeric",
                        f"Daily unique songs / plays, last {q} days. NULL if no plays.",
                        "Exploration vs repeat listening.", *LOGS),
        ]
    out += [
        FeatureSpec("skip_rate_change_30_60", "listening quality", "numeric",
                    "Skip rate in the last 30 days minus skip rate in days 31-90.",
                    "Rising skipping relative to the user's own baseline.", *LOGS),
        FeatureSpec("completion_rate_change_30_60", "listening quality", "numeric",
                    "Completion rate in the last 30 days minus completion rate in days 31-90.",
                    "Falling satisfaction relative to the user's own baseline.", *LOGS),
    ]
    return out


EXPIRY = ("medium", "Expiry dates are written on pre-cutoff transactions, but an export can "
                    "restate them later. Mitigated by the v1_then_v2 source policy "
                    "(Phase 1, decision #12).")


def transaction_specs(cfg: dict[str, Any]) -> list[FeatureSpec]:
    """Plan, price, discount, renewal, cancellation and payment features."""
    lb, pb = cfg["transaction_lookback_days"], cfg["promo_lookback_days"]
    last = "the latest non-cancel transaction before the cutoff"
    return [
        FeatureSpec("last_plan_days", "plan", "numeric", f"payment_plan_days of {last}.",
                    "Plan length; short trial-like plans churn far more (Phase 3).", *LOW),
        FeatureSpec("last_list_price", "plan", "numeric", f"plan_list_price (NTD) of {last}.",
                    "Price tier of the current plan.", *LOW),
        FeatureSpec("last_amount_paid", "plan", "numeric", f"actual_amount_paid (NTD) of {last}.",
                    "What the user actually pays.", *LOW),
        FeatureSpec("last_discount_amount", "plan", "numeric",
                    f"plan_list_price - actual_amount_paid of {last}.",
                    "Size of any discount on the current plan.", *LOW),
        FeatureSpec("last_price_per_day", "plan", "numeric",
                    f"actual_amount_paid / payment_plan_days of {last}.",
                    "Normalises price across plan lengths.", *LOW),
        FeatureSpec("last_is_discounted", "plan", "binary",
                    f"1 if 0 < paid < list price on {last}.", "Promotional pricing flag.", *LOW),
        FeatureSpec("last_is_free", "plan", "binary", f"1 if amount paid is 0 on {last}.",
                    "Free periods are associated with much higher churn (Phase 3).", *LOW),
        FeatureSpec("last_list_price_zero", "plan", "binary",
                    f"1 if list price is 0 on {last}.", "Zero-price plans (trials, bundles).",
                    *LOW),
        FeatureSpec("last_is_auto_renew", "plan", "binary", f"is_auto_renew of {last}.",
                    "Strongest single association with churn in Phase 3 (V 0.38).", *LOW),
        FeatureSpec("last_payment_method", "plan", "categorical",
                    f"payment_method_id of {last}, as a category.",
                    "Payment channel; strongly associated with churn in Phase 3 (V 0.38).",
                    *LOW),
        FeatureSpec("last_txn_is_cancel", "cancellation", "binary",
                    "1 if the most recent pre-cutoff transaction of any kind is a cancellation.",
                    "An active cancellation before the cutoff.", *LOW),
        FeatureSpec("days_to_expiry", "plan", "numeric",
                    "Latest membership_expire_date over non-cancel pre-cutoff transactions, "
                    "minus the cutoff (days).", "How soon the membership runs out.", *EXPIRY),
        FeatureSpec("days_to_latest_txn_expiry", "plan", "numeric",
                    "Expiry on the most recent pre-cutoff transaction (cancellations included), "
                    "minus the cutoff.", "Captures expiries moved earlier by a cancellation.",
                    *EXPIRY),
        FeatureSpec(f"n_transactions_{lb}d", "history", "numeric",
                    f"Non-cancel transactions in the last {lb} days.", "Payment activity.", *LOW),
        FeatureSpec(f"total_paid_{lb}d", "history", "numeric",
                    f"NTD paid (non-cancel) in the last {lb} days.", "Customer value to date.",
                    *LOW),
        FeatureSpec(f"share_auto_renew_{lb}d", "history", "numeric",
                    f"Share of non-cancel transactions in the last {lb} days with auto-renew.",
                    "Stability of the renewal habit.", *LOW),
        FeatureSpec(f"n_plan_changes_{lb}d", "history", "numeric",
                    f"Consecutive non-cancel transactions in the last {lb} days whose plan "
                    "length or list price differs from the previous one.",
                    "Plan switching (up/downgrades).", *LOW),
        FeatureSpec(f"n_payment_method_changes_{lb}d", "history", "numeric",
                    f"Consecutive non-cancel transactions in the last {lb} days with a "
                    "different payment method from the previous one.",
                    "Payment instability.", *LOW),
        FeatureSpec(f"n_promo_txns_{pb}d", "history", "numeric",
                    f"Non-cancel transactions in the last {pb} days with list price 0 or "
                    "paid below list.", "Reliance on promotions.", *LOW),
        FeatureSpec("n_cancels_all", "cancellation", "numeric",
                    "Cancellation transactions before the cutoff (history from 2015-01-01).",
                    "Past cancellations.", *LOW),
        FeatureSpec(f"n_cancels_{lb}d", "cancellation", "numeric",
                    f"Cancellation transactions in the last {lb} days.", "Recent cancellations.",
                    *LOW),
        FeatureSpec("days_since_last_cancel", "cancellation", "numeric",
                    "Days since the latest pre-cutoff cancellation; NULL if none.",
                    "Recency of cancelling.", *LOW),
        FeatureSpec("days_since_last_transaction", "history", "numeric",
                    "Days since the latest pre-cutoff transaction of any kind.",
                    "Payment recency.", *LOW),
        FeatureSpec("days_since_first_transaction", "history", "numeric",
                    "Days since the earliest pre-cutoff transaction (left-censored: history "
                    "starts 2015-01-01).", "Length of observed paying history.", *LOW),
        FeatureSpec("no_transaction_history", "history", "binary",
                    "1 if the user has no pre-cutoff transaction at all.",
                    "Missing history is itself informative (Phase 3: 33.8% churn).", *LOW),
    ]


MEMBER = ("medium", "members_v3 is one undated extract: attributes may have changed after "
                    "the cutoff. Users who registered on/after the cutoff are treated exactly "
                    "like users with no members row, so post-cutoff registration is not "
                    "revealed.")


def member_specs(cfg: dict[str, Any]) -> list[FeatureSpec]:
    """Tenure and demographics (demographics excluded from the model by default)."""
    demo = set(cfg["demographics"]["features"])
    use_demo = bool(cfg["demographics"]["use_in_model"])
    return [
        FeatureSpec("tenure_days", "tenure", "numeric",
                    "Days from registration to the cutoff; NULL if unknown.",
                    "Longer-tenured users may be more committed.", *MEMBER),
        FeatureSpec("registered_via", "tenure", "categorical", "Registration channel code.",
                    "Acquisition channel; associated with churn in Phase 3 (V 0.17).", *MEMBER),
        FeatureSpec("member_info_missing", "tenure", "binary",
                    "1 if no usable members row (none, or registered on/after the cutoff).",
                    "Missing profile information.", *MEMBER),
        FeatureSpec("age", "demographics", "numeric",
                    "Age in years if 10-90, else NULL (Phase 1 rule M1).",
                    "Built for the fairness audit.", *MEMBER, use_demo or "age" not in demo),
        FeatureSpec("gender", "demographics", "categorical",
                    "male / female / unknown.", "Built for the fairness audit.", *MEMBER,
                    use_demo or "gender" not in demo),
        FeatureSpec("city", "demographics", "categorical", "City code.",
                    "Built for the fairness audit; can act as a proxy for protected traits.",
                    *MEMBER, use_demo or "city" not in demo),
    ]


RFM = ("low", "Phase 4 scores use pre-cutoff snapshot data, with cut-points and clusters "
              "fitted on snapshot A only and applied unchanged to B.")


def rfm_specs() -> list[FeatureSpec]:
    """Phase 4 RFM outputs reused as features."""
    return [
        FeatureSpec("r_score", "RFM", "numeric", "RFM recency score 1-5 (Phase 4).",
                    "Binned recency on the reference scale.", *RFM),
        FeatureSpec("f_score", "RFM", "numeric", "RFM frequency score 1-5.",
                    "Binned frequency.", *RFM),
        FeatureSpec("m_score", "RFM", "numeric", "RFM monetary score 1-5.",
                    "Binned spend.", *RFM),
        FeatureSpec("personal_recency", "RFM", "numeric",
                    "Current gap / the user's own median gap between active days (needs 5+ "
                    "active days).", "Silence relative to the user's own routine.", *RFM),
        FeatureSpec("rfm_segment", "RFM", "categorical", "Rule-based RFM segment (Phase 4).",
                    "Interpretable engagement segment.", *RFM),
        FeatureSpec("rfm_group", "RFM", "categorical",
                    "K-Means cluster for users who paid and listened; rule label otherwise.",
                    "Captures low-payment listeners (cluster 3: 15.2% churn).", *RFM),
    ]


def all_specs(cfg: dict[str, Any]) -> list[FeatureSpec]:
    """Every feature, in output column order."""
    specs = listening_specs(cfg) + transaction_specs(cfg) + member_specs(cfg) + rfm_specs()
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names in registry")
    return specs
