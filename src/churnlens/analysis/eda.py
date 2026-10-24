"""Phase 3 runner: EDA and behavioural analytics on snapshot A.

Run with `python -m churnlens.analysis.eda` (or `make eda`). Writes figures to
reports/figures/, tables to reports/eda/, and the `eda` section of results.md.
Every statement it writes is an association observed in this data, not a cause.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churnlens.analysis import cohorts, segments, survival
from churnlens.analysis.plotting import (
    CHURNED,
    KEPT,
    SEQUENTIAL,
    SERIES,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    apply_style,
    finish,
    rate_bars,
)
from churnlens.analysis.stats import fmt_p
from churnlens.config import load_config, resolve_path
from churnlens.reporting import fmt_int, md_table, run_banner, write_section

KM_TOP_GROUPS = 4
# Cell share above which the blue ramp is dark enough to need white text.
HEATMAP_WHITE_TEXT_ABOVE = 0.45
TRAJECTORY_METRICS = [
    ("minutes_per_user", "Listening minutes per user per day"),
    ("active_share", "Share of users active that day"),
    ("unique_songs_per_user", "Unique songs per user per day"),
    ("skip_rate", "Skip rate (share of plays < 25% of song)"),
    ("completion_rate", "Completion rate (share of plays > 98.5%)"),
]


def pct(x: float, digits: int = 1) -> str:
    """Fraction -> percent string."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.{digits}f}%"


def median_text(days: float) -> str:
    """Median survival in days, or 'not reached' when the curve never falls below 0.5."""
    return f"{days:.0f}" if math.isfinite(days) else "not reached"


# ---------------------------------------------------------------- figures


def plot_segments(groups: pd.DataFrame, tests: pd.DataFrame, overall: float, prov: str,
                  path: Path, dpi: int) -> Path:
    """Small multiples: churn rate with 95% CI per group, one panel per dimension."""
    order = list(tests["dimension"])
    ncols = 3
    nrows = math.ceil(len(order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.1 * nrows))
    for ax, dim in zip(axes.flat, order, strict=False):
        g = groups[groups["dimension"] == dim].sort_values("order")
        v = tests.loc[tests["dimension"] == dim, "cramers_v"].iloc[0]
        rate_bars(ax, list(g["group"]), list(g["rate"]), list(g["ci_lo"]), list(g["ci_hi"]),
                  list(g["n"]), overall, f"{g['title'].iloc[0]}  (Cramér's V {v:.2f})")
    for ax in list(axes.flat)[len(order):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return finish(fig, f"Churn rate by segment (dashed line = overall {pct(overall)}; "
                       "whiskers = 95% CI)", prov, path, dpi)


def plot_trajectories(traj: pd.DataFrame, smooth: int, prov: str, path: Path, dpi: int) -> Path:
    """Two-series lines (retained vs churned) for each listening metric, 90 days to cutoff."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (col, title) in zip(axes.flat, TRAJECTORY_METRICS, strict=False):
        for label, churn, colour in (("retained", 0, KEPT), ("churned", 1, CHURNED)):
            s = traj[traj["is_churn"] == churn].set_index("day")[col]
            s = s.rolling(smooth, min_periods=1).mean()
            scale = 100 if "rate" in col or "share" in col else 1
            ax.plot(s.index, s.values * scale, color=colour, label=label)
            ax.text(s.index[-1] + 1, s.values[-1] * scale, label, color=TEXT_PRIMARY,
                    fontsize=8, va="center")
        ax.set_title(title + (" (%)" if "rate" in col or "share" in col else ""), loc="left")
        ax.set_xlabel("days before cutoff")
        ax.set_xlim(traj["day"].min(), 8)
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    for i, (label, colour) in enumerate((("retained (is_churn = 0)", KEPT),
                                         ("churned (is_churn = 1)", CHURNED))):
        legend_ax.plot([0.05, 0.2], [0.7 - 0.15 * i] * 2, color=colour,
                       transform=legend_ax.transAxes)
        legend_ax.text(0.25, 0.7 - 0.15 * i, label, transform=legend_ax.transAxes, va="center")
    legend_ax.text(0.05, 0.3, f"{smooth}-day rolling mean (display only).\nPer-user values count "
                   "inactive days as 0.", transform=legend_ax.transAxes, color=TEXT_SECONDARY,
                   fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return finish(fig, "Listening in the 90 days before the cutoff: churned vs retained users",
                  prov, path, dpi)


def plot_cohort_heatmap(matrix: pd.DataFrame, prov: str, path: Path, dpi: int) -> Path:
    """Cohort (registration month) x months since registration -> active share."""
    fig, ax = plt.subplots(figsize=(11, 0.38 * len(matrix) + 1.8))
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(data) * 100, cmap=SEQUENTIAL, aspect="auto", vmin=0,
                   vmax=100)
    for (i, j), v in np.ndenumerate(data):
        if not math.isnan(v):
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > HEATMAP_WHITE_TEXT_ABOVE else TEXT_PRIMARY)
    ax.set_xticks(range(data.shape[1]), [str(k) for k in matrix.columns])
    ax.set_yticks(range(data.shape[0]), [pd.Timestamp(c).strftime("%Y-%m") for c in matrix.index])
    ax.set_xlabel("months since registration month")
    ax.set_ylabel("registration cohort")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="share with an active membership (%)", shrink=0.8)
    return finish(fig, "Cohort activity: share of each registration cohort with an active "
                       "membership", prov, path, dpi)


def plot_channel_curves(curves: pd.DataFrame, prov: str, path: Path, dpi: int) -> Path:
    """Active share by months since registration for the largest channels."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = curves.groupby("registered_via")["cohort_users"].first().sort_values(ascending=False)
    for colour, channel in zip(SERIES, order.index, strict=False):
        c = curves[curves["registered_via"] == channel].sort_values("k")
        label = f"channel {channel} (n={int(order[channel]):,})"
        ax.plot(c["k"], c["share"] * 100, color=colour, marker="o", markersize=4, label=label)
        ax.text(c["k"].iloc[-1] + 0.2, c["share"].iloc[-1] * 100, f"channel {channel}",
                fontsize=8, va="center", color=TEXT_PRIMARY)
    ax.set_xlabel("months since registration month")
    ax.set_ylabel("active share (%)")
    ax.legend(loc="lower left", fontsize=8)
    return finish(fig, "Cohort activity by registration channel (pooled cohorts)", prov, path, dpi)


def plot_km(results: dict[str, dict[str, Any]], max_days: int, prov: str, path: Path,
            dpi: int) -> Path:
    """Kaplan–Meier survival of subscription spells, one panel per grouping."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, res in zip(axes.flat, results.values(), strict=False):
        for colour, (group, kmf) in zip(SERIES, res["curves"].items(), strict=False):
            sf = kmf.survival_function_.iloc[:, 0]
            ci = kmf.confidence_interval_
            ax.step(sf.index, sf.values, where="post", color=colour,
                    label=f"{group} (n={len(kmf.durations):,})")
            ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post", color=colour,
                            alpha=0.15, linewidth=0)
        ax.set_xlim(0, max_days)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{res['title']}  (log-rank p {fmt_p(res['logrank_p'])})", loc="left")
        ax.set_xlabel("days since spell start")
        ax.set_ylabel("share of spells not yet churned")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return finish(fig, "Kaplan–Meier survival of subscription spells (bands = 95% CI)", prov,
                  path, dpi)


def plot_promo(promo: pd.DataFrame, overall: float, prov: str, path: Path, dpi: int) -> Path:
    """Churn rate after promotional vs full-price periods."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    rate_bars(ax, list(promo["group"]), list(promo["rate"]), list(promo["ci_lo"]),
              list(promo["ci_hi"]), list(promo["n"]), overall, "")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    return finish(fig, "Churn after promotional vs full-price periods", prov, path, dpi)


# ---------------------------------------------------------------- observations


def observations(groups: pd.DataFrame, tests: pd.DataFrame, summary: pd.DataFrame,
                 promo: pd.DataFrame, km: dict[str, dict[str, Any]], channel: pd.DataFrame,
                 overall: float) -> list[str]:
    """Plain-language findings, each built from the numbers above (association wording)."""
    out = []
    # Extremes are taken over substantive groups only, not catch-alls or missing-data rows.
    catch_all = {segments.OTHER, "no transaction", "no members row", "unknown",
                 "unknown / invalid", "out of range"}
    for rank, (_, t) in enumerate(tests.head(4).iterrows(), 1):
        g = groups[(groups["dimension"] == t["dimension"]) & ~groups["group"].isin(catch_all)]
        hi, lo = g.loc[g["rate"].idxmax()], g.loc[g["rate"].idxmin()]
        strength = "the strongest" if rank == 1 else f"the #{rank}"
        out.append(f"**{t['title']}** has {strength} association with churn of the "
                   f"{len(tests)} dimensions tested (Cramér's V {t['cramers_v']:.2f}): churn "
                   f"ranges from {pct(lo['rate'])} for *{lo['group']}* (n={lo['n']:,}) to "
                   f"{pct(hi['rate'])} for *{hi['group']}* (n={hi['n']:,}), vs {pct(overall)} "
                   "overall.")
    kept, churned = summary.set_index("is_churn").loc[0], summary.set_index("is_churn").loc[1]
    out.append(
        "In the 30 days before the cutoff, users who went on to churn listened a median "
        f"{churned['median_recent_min_per_day']:.1f} min/day vs "
        f"{kept['median_recent_min_per_day']:.1f} for retained users; their median "
        f"recent-to-earlier listening ratio was {churned['median_recent_to_earlier_ratio']:.2f} "
        f"vs {kept['median_recent_to_earlier_ratio']:.2f}, and "
        f"{pct(churned['share_down_over_half'])} vs {pct(kept['share_down_over_half'])} more "
        "than halved their listening. Lower and declining listening is associated with churn, "
        "but the gap is modest at the median.")
    any_promo = promo[promo["group"].str.contains("any promotional")].iloc[0]
    full = promo[promo["group"] == "last plan: full price"].iloc[0]
    out.append(f"Users whose last plan was promotional (free, list price 0, or discounted; "
               f"n={any_promo['n']:,}) churned at {pct(any_promo['rate'])} vs "
               f"{pct(full['rate'])} after a full-price plan (p {fmt_p(any_promo['p_value'])}). "
               "This is an association: promotions may attract users who were already less "
               "committed.")
    for key in ("auto_renew", "first_payment"):
        t = km[key]["table"].set_index("group")
        parts = [f"*{g}* {median_text(r['median_days'])}" for g, r in t.iterrows()]
        out.append(f"Spell survival by {km[key]['title'].lower()}: median days to churn "
                   f"{', '.join(parts)} (log-rank p {fmt_p(km[key]['logrank_p'])}).")
    last_k = int(channel["k"].max())
    end = channel[channel["k"] == last_k].sort_values("share")
    out.append(f"{last_k} months after registration, the active share ranges from "
               f"{pct(end['share'].iloc[0])} (channel {end['registered_via'].iloc[0]}) to "
               f"{pct(end['share'].iloc[-1])} (channel {end['registered_via'].iloc[-1]}) across "
               "the largest registration channels.")
    return out


# ---------------------------------------------------------------- results section


def render_section(data_cfg: dict[str, Any], cfg: dict[str, Any], snap: str, cutoff: dt.date,
                   n_users: int, overall: float, groups: pd.DataFrame, tests: pd.DataFrame,
                   summary: pd.DataFrame, promo: pd.DataFrame, matrix: pd.DataFrame,
                   channel: pd.DataFrame, km: dict[str, dict[str, Any]], n_spells: int,
                   spell_event_rate: float, figures: dict[str, Path], obs: list[str]) -> str:
    """Markdown for the `eda` section of results.md."""
    rel = {k: p.relative_to(resolve_path("reports", data_cfg)).as_posix()
           for k, p in figures.items()}
    parts = ["## Phase 3: EDA and behavioural analytics", "", run_banner(data_cfg), "",
             f"Snapshot **{snap}** only (cutoff {cutoff}, {fmt_int(n_users)} users, churn "
             f"{pct(overall, 2)}); snapshot B is held out. All findings are associations in "
             "observational data, not causal effects.", "", "### Key observations", ""]
    parts += [f"{i}. {o}" for i, o in enumerate(obs, 1)]

    parts += ["", "### Churn rate by segment", "", f"![segments]({rel['segments']})", "",
              "Association strength per dimension (chi-square test of independence on all "
              "raw categories):", ""]
    parts.append(md_table(["Dimension", "Groups", "χ²", "p", "Cramér's V"],
                          [[t.title, t.groups_tested, f"{t.chi2:,.0f}", fmt_p(t.p_value),
                            f"{t.cramers_v:.3f}"] for t in tests.itertuples()], "lrrrr"))
    parts += ["", f"Groups (categories under {cfg['min_group_size']} users or beyond the "
                  f"{cfg['max_categories']} largest are folded into Other):", ""]
    order = {d: i for i, d in enumerate(tests["dimension"])}
    g = groups.assign(o=groups["dimension"].map(order)).sort_values(["o", "order"])
    parts.append(md_table(["Dimension", "Group", "Users", "Churners", "Churn rate", "95% CI"],
                          [[r.title, r.group, fmt_int(r.n), fmt_int(r.churners), pct(r.rate),
                            f"{pct(r.ci_lo)}–{pct(r.ci_hi)}"] for r in g.itertuples()],
                          "llrrrl"))

    parts += ["", "### Listening trajectories (90 days before cutoff)", "",
              f"![trajectories]({rel['trajectories']})", ""]
    parts.append(md_table(
        ["Group", "Users", "Median min/day, last 30d", "Median min/day, days 31–90",
         "Median recent/earlier ratio", "Share that more than halved", "Median active days, "
         "last 30d", "Share inactive last 30d"],
        [["churned" if r.is_churn else "retained", fmt_int(r.users),
          f"{r.median_recent_min_per_day:.1f}", f"{r.median_earlier_min_per_day:.1f}",
          f"{r.median_recent_to_earlier_ratio:.2f}", pct(r.share_down_over_half),
          f"{r.median_recent_active_days:.0f}", pct(r.share_inactive_recent)]
         for r in summary.itertuples()], "lrrrrrrr"))

    parts += ["", "### Discounts and promotions", "", f"![promo]({rel['promo']})", ""]
    parts.append(md_table(["Group", "Users", "Churn rate", "95% CI", "z vs reference", "p"],
                          [[r.group, fmt_int(r.n), pct(r.rate), f"{pct(r.ci_lo)}–{pct(r.ci_hi)}",
                            "—" if math.isnan(r.z_vs_reference) else f"{r.z_vs_reference:.1f}",
                            fmt_p(r.p_value)] for r in promo.itertuples()], "lrrlrr"))
    parts.append("\nReferences: last-plan rows vs *full price*; recent-promo row vs *no promo*.")

    c = cfg["cohorts"]
    parts += ["", "### Cohort activity by registration month and channel", "",
              f"Members registered {c['first_month']} to {c['last_month']} with at least one "
              "transaction before the cutoff; share with an active membership at each month end "
              "(cancellations not netted out).", "", f"![cohorts]({rel['cohorts']})", "",
              f"![channels]({rel['channels']})", ""]
    ks = [k for k in (0, 1, 3, 6, 12) if k in channel["k"].unique()]
    rows = []
    for ch, sub in channel.groupby("registered_via"):
        s = sub.set_index("k")["share"]
        rows.append([f"channel {ch}", fmt_int(int(sub["cohort_users"].iloc[0])),
                     *[pct(s.get(k, float("nan"))) for k in ks]])
    parts.append(md_table(["Channel", "Users", *[f"k={k}" for k in ks]], rows,
                          "lr" + "r" * len(ks)))

    s = cfg["survival"]
    parts += ["", "### Survival of subscription spells (Kaplan–Meier)", "",
              f"{fmt_int(n_spells)} spells from all sampled users' transactions before the "
              f"cutoff; a spell churns when no renewal follows within {s['gap_days']} days of its "
              f"end ({pct(spell_event_rate)} of spells observed to churn; the rest censored). "
              f"Spells first seen before {s['min_first_spell_start']} are excluded (left "
              "truncation).", "", f"![km]({rel['km']})", ""]
    rows = []
    for res in km.values():
        for _, r in res["table"].iterrows():
            surv = [pct(r[f"S({d}d)"]) for d in s["report_days"]]
            rows.append([res["title"], r["group"], fmt_int(r["spells"]), fmt_int(r["events"]),
                         median_text(r["median_days"]), *surv, fmt_p(res["logrank_p"])])
    parts.append(md_table(["Grouping", "Group", "Spells", "Churn events", "Median days",
                           *[f"S({d}d)" for d in s["report_days"]], "Log-rank p"], rows,
                          "llrrr" + "r" * len(s["report_days"]) + "r"))
    return "\n".join(parts)


def overlapping_events(cutoff: str, recent: int, events: list[dict[str, Any]]) -> list[str]:
    """Names of configured calendar events that overlap the last `recent` days before cutoff."""
    end = dt.date.fromisoformat(cutoff)
    start = end - dt.timedelta(days=recent)
    return [e["name"] for e in events if e["start"] < end and e["end"] >= start]


def render_calendar(calendar: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[str, str]:
    """Markdown table for the calendar check, plus one observation sentence."""
    cc = cfg["calendar_check"]
    rows, flagged = [], []
    for c in calendar:
        events = overlapping_events(c["cutoff"], cc["recent_days"], cc["events"])
        if events:
            flagged.append((c, events))
        rows.append([c["cutoff"], f"{c['recent_active_users']:,.0f}",
                     f"{c['baseline_active_users']:,.0f}", f"{c['active_change'] * 100:+.1f}%",
                     f"{c['recent_min_per_active']:.1f}", f"{c['baseline_min_per_active']:.1f}",
                     f"{c['minutes_change'] * 100:+.1f}%", ", ".join(events) or "none"])
    table = "\n".join([
        "### Calendar check (all sampled users, before each cutoff)", "",
        f"Last {cc['recent_days']} days before each cutoff vs the {cc['baseline_days']} days "
        "before that. No labels involved; this describes the calendar, not churn.", "",
        md_table(["Cutoff", f"Active users/day, last {cc['recent_days']}d", "Baseline",
                  "Change", "Min per active user, last 7d", "Baseline", "Change",
                  "Configured events overlapping"], rows, "lrrrrrrl")])
    obs = ""
    if flagged:
        c, events = flagged[0]
        obs = (f"All-user activity in the {cc['recent_days']} days before the {c['cutoff']} "
               f"cutoff differs from the prior {cc['baseline_days']} days by "
               f"{c['active_change'] * 100:+.1f}% (active users) and "
               f"{c['minutes_change'] * 100:+.1f}% (minutes per active user); that window "
               f"overlaps the {', '.join(events)}. This timing coincidence is noted, not tested. "
               "Short-window listening features will differ between snapshots for calendar "
               "reasons as well as behavioural ones.")
    return table, obs


# ---------------------------------------------------------------- entry point


def main() -> int:
    """Run every Phase 3 analysis and write figures, tables and the results section."""
    data_cfg, cfg = load_config("data"), load_config("eda")
    snap_cfg = load_config("snapshots")
    snap = cfg["snapshot"]
    cutoff: dt.date = snap_cfg["snapshots"][snap]["cutoff"]
    snap_dir = resolve_path(snap_cfg["outputs"]["snapshot_dir"], data_cfg) / snap
    fig_dir = resolve_path(cfg["figures"]["dir"], data_cfg)
    tab_dir = resolve_path(cfg["outputs"]["tables_dir"], data_cfg)
    tab_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg["figures"]["dpi"])
    pct_s = 100 if data_cfg["mode"] == "full" else data_cfg["user_sample_pct"]
    prov = (f"Source: KKBox, mode {data_cfg['mode']}, {pct_s}% user sample (seed "
            f"{data_cfg['sample_seed']}), snapshot {snap} (cutoff {cutoff}). Associations only.")
    apply_style()

    db = resolve_path(load_config("cleaning")["outputs"]["database"], data_cfg)
    con = duckdb.connect(str(db), read_only=True)
    segments.build_base(con, snap_dir, cutoff)
    n_users, n_churn = con.execute("SELECT COUNT(*), SUM(is_churn) FROM base").fetchone()
    overall = n_churn / n_users
    groups, tests = segments.segment_rates(con, cfg)
    promo = segments.promo_analysis(con, snap_dir, cutoff, cfg)
    traj = segments.trajectories(con, snap_dir, cutoff, int(cfg["trajectory_days"]))
    summary = segments.trajectory_summary(con, snap_dir, cutoff, int(cfg["trajectory_days"]),
                                          int(cfg["recent_days"]))
    cc = cfg["calendar_check"]
    calendar = [segments.calendar_check(con, s["cutoff"], int(cc["recent_days"]),
                                        int(cc["baseline_days"]))
                for s in snap_cfg["snapshots"].values()]
    activity = cohorts.cohort_activity(con, cutoff, cfg["cohorts"])
    matrix = cohorts.cohort_matrix(activity)
    channel = cohorts.channel_curves(activity, int(cfg["cohorts"]["top_channels"]))
    spells = survival.build_spells(con, cutoff, cfg["survival"])
    km = {key: survival.km_by_group(con, spells, key, cfg["survival"], cfg["alpha"],
                                    KM_TOP_GROUPS) for key in survival.GROUPINGS}
    con.close()

    figures = {
        "segments": plot_segments(groups, tests, overall, prov,
                                  fig_dir / "phase3_churn_by_segment.png", dpi),
        "trajectories": plot_trajectories(traj, int(cfg["trajectory_smoothing_days"]), prov,
                                          fig_dir / "phase3_listening_trajectories.png", dpi),
        "promo": plot_promo(promo, overall, prov, fig_dir / "phase3_promotions.png", dpi),
        "cohorts": plot_cohort_heatmap(matrix, prov, fig_dir / "phase3_cohort_heatmap.png", dpi),
        "channels": plot_channel_curves(channel, prov, fig_dir / "phase3_cohort_channels.png",
                                        dpi),
        "km": plot_km(km, int(cfg["survival"]["plot_max_days"]), prov,
                      fig_dir / "phase3_kaplan_meier.png", dpi),
    }
    for name, frame in {"segment_rates": groups, "segment_tests": tests, "promotions": promo,
                        "trajectories_daily": traj, "trajectory_summary": summary,
                        "cohort_activity": activity, "cohort_channels": channel,
                        "km_summary": pd.concat([r["table"] for r in km.values()])}.items():
        # 10 significant digits: parallel float sums differ in the last bits run to run.
        frame.to_csv(tab_dir / f"{name}.csv", index=False, float_format="%.10g")

    obs = observations(groups, tests, summary, promo, km, channel, overall)
    calendar_md, calendar_obs = render_calendar(calendar, cfg)
    if calendar_obs:
        obs.append(calendar_obs)
    pd.DataFrame(calendar).to_csv(tab_dir / "calendar_check.csv", index=False,
                                  float_format="%.10g")
    section = render_section(data_cfg, cfg, snap, cutoff, n_users, overall, groups, tests,
                             summary, promo, matrix, channel, km, len(spells),
                             float(spells["event"].mean()), figures, obs)
    write_section(resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg), "eda",
                  f"{section}\n\n{calendar_md}")
    print(f"phase 3: {len(figures)} figures -> {fig_dir}, tables -> {tab_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
