"""Phase 4 runner: RFM scores, rule-based segments, optional K-Means, report.

Run with `python -m churnlens.segmentation.run` (or `make rfm`).
  - fits score references (and K-Means) on the reference snapshot (A) only
  - scores every snapshot with those references -> data/processed/{mode}/rfm/{S}.parquet
  - reports churn by segment for the reference snapshot only (B stays held out;
    for B only segment shares are shown, which use no labels)
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churnlens.analysis.plotting import (
    SEQUENTIAL,
    SERIES,
    TEXT_PRIMARY,
    apply_style,
    finish,
    rate_bars,
)
from churnlens.analysis.stats import chi2_independence, wilson
from churnlens.config import load_config, resolve_path
from churnlens.reporting import fmt_int, md_table, run_banner, write_section
from churnlens.segmentation import kmeans, rfm

HEATMAP_WHITE_TEXT_ABOVE = 0.45


def pct(x: float) -> str:
    """Fraction -> percent with one decimal."""
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def sig10(values: np.ndarray) -> list:
    """Array -> nested list rounded to 10 significant digits (stable across runs)."""
    return np.vectorize(lambda v: float(f"{v:.10g}"))(np.asarray(values)).tolist()


def rate_table(df: pd.DataFrame, by: str, alpha: float) -> pd.DataFrame:
    """Churn rate with Wilson CI and median RFM values per group."""
    rows = []
    for g, sub in df.groupby(by, sort=False):
        r = wilson(int(sub["is_churn"].sum()), len(sub), alpha)
        rows.append({by: g, "n": len(sub), "share": len(sub) / len(df), "churners": r.k,
                     "rate": r.rate, "ci_lo": r.lo, "ci_hi": r.hi,
                     "median_recency": sub["recency_days"].median(),
                     "median_frequency": sub["frequency_days"].median(),
                     "median_monetary": sub["monetary"].median()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures


def plot_segments(seg: pd.DataFrame, overall: float, prov: str, path: Path, dpi: int) -> Path:
    """Churn rate by RFM segment with 95% CI."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rate_bars(ax, list(seg["segment"]), list(seg["rate"]), list(seg["ci_lo"]),
              list(seg["ci_hi"]), list(seg["n"]), overall, "")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    return finish(fig, "Churn rate by RFM segment (rule order; dashed = overall)", prov, path,
                  dpi)


def plot_rf_heatmap(df: pd.DataFrame, prov: str, path: Path, dpi: int) -> Path:
    """Churn rate (and n) for each R score x F score cell."""
    agg = df.groupby(["r_score", "f_score"])["is_churn"].agg(["mean", "size"]).reset_index()
    rate = agg.pivot(index="r_score", columns="f_score", values="mean").sort_index(ascending=False)
    size = agg.pivot(index="r_score", columns="f_score", values="size").reindex_like(rate)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    data = rate.to_numpy(dtype=float)
    vmax = float(np.nanmax(data))
    im = ax.imshow(np.ma.masked_invalid(data) * 100, cmap=SEQUENTIAL, aspect="auto", vmin=0,
                   vmax=vmax * 100)
    for (i, j), v in np.ndenumerate(data):
        if not math.isnan(v):
            ax.text(j, i, f"{v * 100:.1f}%\nn={int(size.iat[i, j]):,}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if v / vmax > HEATMAP_WHITE_TEXT_ABOVE else TEXT_PRIMARY)
    ax.set_xticks(range(rate.shape[1]), [str(c) for c in rate.columns])
    ax.set_yticks(range(rate.shape[0]), [str(r) for r in rate.index])
    ax.set_xlabel("F score (active days, 5 = most)")
    ax.set_ylabel("R score (5 = most recent)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="churn rate (%)", shrink=0.85)
    return finish(fig, "Churn rate by recency and frequency score", prov, path, dpi)


def plot_kmeans_selection(diag: pd.DataFrame, k: int, prov: str, path: Path, dpi: int) -> Path:
    """Elbow (inertia) and silhouette side by side: two charts, never two y-axes."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, title in ((a1, "inertia", "Inertia (elbow)"),
                           (a2, "silhouette", "Silhouette score (sampled)")):
        ax.plot(diag["k"], diag[col], color=SERIES[0], marker="o", markersize=5)
        chosen = diag[diag["k"] == k]
        ax.plot(chosen["k"], chosen[col], marker="o", markersize=9, color=SERIES[1],
                linestyle="none")
        ax.annotate(f"chosen k={k}", (k, float(chosen[col].iloc[0])), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, color=TEXT_PRIMARY)
        ax.set_title(title, loc="left")
        ax.set_xlabel("k")
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    return finish(fig, "K-Means model selection on scaled RFM (k chosen by silhouette)", prov,
                  path, dpi)


def plot_crosstab(ct: pd.DataFrame, cluster_churn: pd.Series, prov: str, path: Path,
                  dpi: int) -> Path:
    """Share of each RFM segment falling in each cluster (rows sum to 100%)."""
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(ct) + 1.8))
    data = ct.to_numpy(dtype=float)
    im = ax.imshow(data * 100, cmap=SEQUENTIAL, aspect="auto", vmin=0, vmax=100)
    for (i, j), v in np.ndenumerate(data):
        ax.text(j, i, f"{v * 100:.0f}%", ha="center", va="center", fontsize=8,
                color="white" if v > HEATMAP_WHITE_TEXT_ABOVE else TEXT_PRIMARY)
    ax.set_xticks(range(ct.shape[1]),
                  [f"{c}\nchurn {cluster_churn[c] * 100:.1f}%" for c in ct.columns])
    ax.set_yticks(range(ct.shape[0]), list(ct.index))
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="share of segment (%)", shrink=0.85)
    return finish(fig, "RFM segments vs K-Means clusters, clustered users only "
                       "(row-normalised)", prov, path, dpi)


# ---------------------------------------------------------------- report


def render_section(data_cfg: dict[str, Any], cfg: dict[str, Any], ref: str, cutoff: dt.date,
                   overall: float, dist: pd.DataFrame, seg: pd.DataFrame, seg_test: tuple,
                   prec: pd.DataFrame, km: dict[str, Any] | None, b_shares: pd.DataFrame,
                   figures: dict[str, Path], refs: dict[str, rfm.ScoreReference]) -> str:
    """Markdown for the `rfm` section of results.md."""
    rel = {k: p.relative_to(resolve_path("reports", data_cfg)).as_posix()
           for k, p in figures.items()}
    parts = ["## Phase 4: RFM and segmentation", "", run_banner(data_cfg), "",
             f"Scores and K-Means are fitted on snapshot **{ref}** (cutoff {cutoff}) and applied "
             "unchanged to other snapshots. Churn by segment is shown for the reference snapshot "
             "only. Associations, not causes.", "",
             f"- **Recency**: days since last listen (no listen in {cfg['recency_cap_days']} days "
             f"= {cfg['recency_cap_days'] + 1}); **Frequency**: active days in the last "
             f"{cfg['frequency_window_days']}; **Monetary**: NTD paid in the last "
             f"{cfg['monetary_window_days']} days (non-cancel transactions).",
             f"- **Scores** 1–{cfg['n_levels']}: ⌈{cfg['n_levels']} × share of reference users "
             "at or below the value⌉ (recency inverted). Tie-aware, so piled-up values leave "
             "some scores unused:", ""]
    parts.append(md_table(["Score", *[str(k) for k in range(1, cfg["n_levels"] + 1)]],
                          [[r["score"], *[pct(r[str(k)]) for k in range(1, cfg["n_levels"] + 1)]]
                           for _, r in dist.iterrows()], "l" + "r" * cfg["n_levels"]))
    parts += ["", "Score cut-points (raw value range per score, reference snapshot):", ""]
    rows = []
    for col, (score_col, sign) in rfm.SCORED.items():
        ref_obj = refs[col]
        for level in range(1, cfg["n_levels"] + 1):
            lo_share, hi_share = (level - 1) / cfg["n_levels"], level / cfg["n_levels"]
            mask = (ref_obj.cum_share > lo_share + rfm.CEIL_TOLERANCE) & \
                   (ref_obj.cum_share <= hi_share + rfm.CEIL_TOLERANCE)
            vals = sign * ref_obj.values[mask]
            rng = "unused" if len(vals) == 0 else (
                f"{vals.min():g}" if vals.min() == vals.max() else f"{vals.min():g}–{vals.max():g}")
            rows.append([score_col, level, rng])
    parts.append(md_table(["Score", "Level", "Raw values"], rows, "lrl"))

    parts += ["", "### Segments (first matching rule wins)", "", f"![segments]({rel['segments']})",
              ""]
    meaning = {r["name"]: (r["when"], r["meaning"]) for r in cfg["segments"]}
    parts.append(md_table(
        ["Precedence", "Segment", "Rule", "Users", "Share", "Churn rate", "95% CI",
         "Median R / F / M"],
        [[i + 1, r["segment"], f"`{meaning[r['segment']][0]}`", fmt_int(r["n"]), pct(r["share"]),
          pct(r["rate"]), f"{pct(r['ci_lo'])}–{pct(r['ci_hi'])}",
          f"{r['median_recency']:g} / {r['median_frequency']:g} / {r['median_monetary']:g}"]
         for i, r in seg.iterrows()], "rllrrrlr"))
    parts.append(f"\nChurn vs segment: χ² {seg_test[0]:,.0f}, Cramér's V **{seg_test[2]:.3f}** "
                 "(compare Phase 3: auto-renew 0.38, payment method 0.38).")
    parts += ["", f"![rf]({rel['rf']})", "", "### Personalised recency", "",
              "Current gap divided by the user's own median gap between active days "
              f"(needs ≥{cfg['personal_recency']['min_active_days']} active days).", ""]
    parts.append(md_table(["Band", "Users", "Churn rate", "95% CI"],
                          [[r["band"], fmt_int(r["n"]), pct(r["rate"]),
                            f"{pct(r['ci_lo'])}–{pct(r['ci_hi'])}"] for _, r in prec.iterrows()],
                          "lrrl"))

    parts += ["", "### K-Means comparison", ""]
    if km is None:
        parts.append("Disabled in `configs/rfm.yaml`.")
    else:
        c = km["compare"]
        verdict = "**kept**" if c["keep"] else "**not kept**"
        parts += [f"![kmeans]({rel['kmeans']})", "",
                  md_table(["k", "Inertia", "Silhouette"],
                           [[int(r.k), f"{r.inertia:,.0f}", f"{r.silhouette:.3f}"]
                            for r in km["diagnostics"].itertuples()], "rrr"), "",
                  f"Clustered population: `{cfg['kmeans']['population']}`: "
                  f"{fmt_int(c['population_users'])} users ({pct(c['population_share'])} of "
                  f"snapshot {ref}, churn {pct(c['population_churn_rate'])}). Everyone else gets a "
                  "rule-based *not clustered* label. Cluster numbers follow size (1 = largest).",
                  "",
                  f"Chosen k = **{km['k']}** (highest silhouette). Within the clustered "
                  f"population: adjusted Rand index vs segments {c['adjusted_rand_index']:.3f}; "
                  f"Cramér's V with churn: segments {c['cramers_v_segments']:.3f}, clusters "
                  f"{c['cramers_v_clusters']:.3f}, segment × cluster "
                  f"{c['cramers_v_segment_x_cluster']:.3f} (gain {c['gain']:+.3f} vs the "
                  f"{cfg['kmeans']['min_cramers_v_gain']} threshold) → K-Means {verdict}.", "",
                  f"![crosstab]({rel['crosstab']})", "",
                  "Cluster and exclusion-group profiles (reference snapshot):", ""]
        parts.append(md_table(
            ["Group", "Users", "Share", "Churn rate", "95% CI", "Median recency",
             "Median active days", "Median paid (NTD)"],
            [[r["cluster"], fmt_int(int(r["n"])), pct(r["share"]), pct(r["rate"]),
              f"{pct(r['ci_lo'])}–{pct(r['ci_hi'])}", f"{r['median_recency']:g}",
              f"{r['median_frequency']:g}", f"{r['median_monetary']:g}"]
             for _, r in km["profiles"].iterrows()], "lrrrlrrr"))
        parts += ["", "Cluster shares by snapshot (B assigned with A's model; no labels used):",
                  ""]
        parts.append(md_table(["Group", *km["shares"].columns],
                              [[i, *[pct(v) for v in r]] for i, r in km["shares"].iterrows()],
                              "l" + "r" * len(km["shares"].columns)))
    parts += ["", "### Segment shares by snapshot (no labels used)", ""]
    parts.append(md_table(["Segment", *b_shares.columns], [[i, *[pct(v) for v in r]]
                                                           for i, r in b_shares.iterrows()],
                          "l" + "r" * len(b_shares.columns)))
    return "\n".join(parts)


# ---------------------------------------------------------------- entry point


def main() -> int:
    """Fit on the reference snapshot, score all snapshots, report."""
    data_cfg, cfg = load_config("data"), load_config("rfm")
    alpha = load_config("eda")["alpha"]
    snap_cfg = load_config("snapshots")
    ref = cfg["reference_snapshot"]
    base = resolve_path(snap_cfg["outputs"]["snapshot_dir"], data_cfg)
    out_dir = resolve_path(cfg["outputs"]["rfm_dir"], data_cfg)
    tab_dir = resolve_path(cfg["outputs"]["tables_dir"], data_cfg)
    fig_dir = resolve_path(cfg["figures"]["dir"], data_cfg)
    for d in (out_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)
    dpi, n_levels = int(cfg["figures"]["dpi"]), int(cfg["n_levels"])
    cutoff = snap_cfg["snapshots"][ref]["cutoff"]
    pct_s = 100 if data_cfg["mode"] == "full" else data_cfg["user_sample_pct"]
    prov = (f"Source: KKBox, mode {data_cfg['mode']}, {pct_s}% user sample (seed "
            f"{data_cfg['sample_seed']}), snapshot {ref} (cutoff {cutoff}). Associations only.")
    apply_style()
    con = duckdb.connect()

    raws = {s: rfm.raw_rfm(con, base / s, spec["cutoff"], cfg)
            for s, spec in snap_cfg["snapshots"].items()}
    refs = rfm.fit_references(raws[ref])
    (tab_dir / "score_reference.json").write_text(
        json.dumps({k: v.to_json() for k, v in refs.items()}), encoding="utf-8")
    km_cfg = cfg["kmeans"]
    km_result = None
    if km_cfg["enabled"]:
        pop_ref = kmeans.in_population(raws[ref], km_cfg["population"])
        km_result = kmeans.select_and_fit(raws[ref][pop_ref], km_cfg)

    scored = {}
    for s, raw in raws.items():
        df = rfm.score(raw, refs, n_levels)
        df["segment"] = rfm.assign_segments(df, cfg["segments"])
        df["personal_recency_band"] = df["personal_recency"].map(rfm.personal_recency_band)
        if km_result is not None:
            df["cluster"] = kmeans.assign(km_result, raw, km_cfg)
            df["clustered"] = kmeans.in_population(raw, km_cfg["population"])
        df.to_parquet(out_dir / f"{s}.parquet", index=False)
        scored[s] = df

    labels = pd.read_parquet(base / ref / "labels.parquet")
    a = scored[ref].merge(labels, on="msno", validate="one_to_one")
    overall = float(a["is_churn"].mean())
    order = [r["name"] for r in cfg["segments"]]
    seg = rate_table(a, "segment", alpha).set_index("segment").reindex(order).dropna(
        subset=["n"]).reset_index()
    seg_test = chi2_independence([(int(r.churners), int(r.n)) for r in seg.itertuples()])
    band_order = ["≤1x usual gap", "1–3x usual gap", "3–10x usual gap", ">10x usual gap",
                  "too little history"]
    prec = rate_table(a, "personal_recency_band", alpha).rename(
        columns={"personal_recency_band": "band"}).set_index("band").reindex(band_order) \
        .dropna(subset=["n"]).reset_index()

    figures = {"segments": plot_segments(seg, overall, prov, fig_dir / "phase4_rfm_segments.png",
                                         dpi),
               "rf": plot_rf_heatmap(a, prov, fig_dir / "phase4_rf_heatmap.png", dpi)}
    km_report = None
    if km_result is not None:
        pop = a[a["clustered"]]
        cmp = kmeans.compare(pop["segment"], pop["cluster"], pop["is_churn"],
                             km_cfg["min_cramers_v_gain"])
        cmp["population_users"] = len(pop)
        cmp["population_share"] = len(pop) / len(a)
        cmp["population_churn_rate"] = float(pop["is_churn"].mean())
        cluster_names = [f"cluster {i}" for i in range(1, km_result.k + 1)]
        label_order = cluster_names + [g["name"] for g in km_cfg["excluded_groups"]]
        prof = rate_table(a, "cluster", alpha).set_index("cluster").reindex(label_order) \
            .dropna(subset=["n"]).reset_index()
        ct = pd.crosstab(pop["segment"], pop["cluster"], normalize="index").reindex(
            index=[o for o in order if o in set(pop["segment"])], columns=cluster_names)
        figures["kmeans"] = plot_kmeans_selection(km_result.diagnostics, km_result.k, prov,
                                                  fig_dir / "phase4_kmeans_selection.png", dpi)
        figures["crosstab"] = plot_crosstab(ct, prof.set_index("cluster")["rate"], prov,
                                            fig_dir / "phase4_segments_vs_clusters.png", dpi)
        km_report = {"k": km_result.k, "diagnostics": km_result.diagnostics, "compare": cmp,
                     "profiles": prof, "shares": pd.DataFrame(
                         {s: df["cluster"].value_counts(normalize=True)
                          for s, df in scored.items()}).reindex(label_order).fillna(0.0)}
        km_result.diagnostics.to_csv(tab_dir / "kmeans_selection.csv", index=False,
                                     float_format="%.10g")
        prof.to_csv(tab_dir / "cluster_profiles.csv", index=False, float_format="%.10g")
        (tab_dir / "kmeans_model.json").write_text(json.dumps({
            "k": km_result.k, "population": km_cfg["population"],
            "label_map": {str(raw): f"cluster {n}" for raw, n in km_result.label_map.items()},
            "features": ["log1p(recency_days)", "frequency_days", "log1p(monetary)"],
            # 10 significant digits: threaded K-Means sums differ in the last bits per run.
            "scaler_mean": sig10(km_result.scaler.mean_),
            "scaler_scale": sig10(km_result.scaler.scale_),
            "centroids": sig10(km_result.model.cluster_centers_),
            **{k: (round(v, 10) if isinstance(v, float) else v) for k, v in cmp.items()}},
            indent=2),
            encoding="utf-8")

    b_shares = pd.DataFrame({s: df["segment"].value_counts(normalize=True)
                             for s, df in scored.items()}).reindex(order).fillna(0.0)
    seg.to_csv(tab_dir / "segment_rates.csv", index=False, float_format="%.10g")
    prec.to_csv(tab_dir / "personal_recency.csv", index=False, float_format="%.10g")
    dist = rfm.score_distribution(scored[ref], n_levels)
    write_section(resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg), "rfm",
                  render_section(data_cfg, cfg, ref, cutoff, overall, dist, seg, seg_test, prec,
                                 km_report, b_shares, figures, refs))
    print(f"phase 4: rfm tables -> {out_dir}, figures -> {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
