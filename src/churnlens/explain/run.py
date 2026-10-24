"""Phase 7 runner: global and local explanations for the Phase 6 model.

Run with `python -m churnlens.explain.run` (or `make explain`). Needs the Phase 6 model
bundle and Phase 5 features. Writes:
  data/processed/{mode}/explain/{S}_shap.parquet     per-user, per-feature SHAP (log-odds)
  data/processed/{mode}/explain/{S}_drivers.parquet  top risk-raising drivers + reason codes
  reports/explain/*.csv, reports/figures/phase7_*.png, the `explain` section of results.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churnlens.analysis.plotting import (
    SEQUENTIAL,
    SERIES,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    apply_style,
    finish,
)
from churnlens.config import load_config, resolve_path
from churnlens.explain import groups, reasons
from churnlens.explain.shap_values import explain
from churnlens.features.registry import all_specs
from churnlens.modeling import data
from churnlens.reporting import fmt_int, md_table, run_banner, write_section

RAISE, LOWER = SERIES[1], SERIES[0]   # orange = raises risk, blue = lowers risk
JITTER = 0.18


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------- figures


def plot_summary(shap_df: pd.DataFrame, X: pd.DataFrame, top: list[str], prov: str,
                 path: Path, dpi: int, seed: int) -> Path:
    """Beeswarm-style summary: SHAP per user for the top features, coloured by value
    percentile within the feature (categoricals in neutral grey)."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(top) + 1.6))
    for i, f in enumerate(reversed(top)):
        s = shap_df[f].to_numpy()
        y = i + rng.uniform(-JITTER, JITTER, len(s))
        v = pd.to_numeric(X[f], errors="coerce") if f in X else pd.Series(np.nan, index=X.index)
        if v.notna().sum() > 1 and v.nunique() > 1:
            colour = SEQUENTIAL(v.rank(pct=True).fillna(0.5).to_numpy())
        else:
            colour = TEXT_SECONDARY
        ax.scatter(s, y, c=colour, s=3, alpha=0.6, linewidths=0, rasterized=True)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_yticks(range(len(top)), list(reversed(top)))
    ax.set_xlabel("SHAP value (log-odds of raw score; > 0 raises predicted risk)")
    ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL)
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_ticks([0, 1], labels=["low value", "high value"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return finish(fig, f"SHAP summary: top {len(top)} features (grey = categorical)", prov,
                  path, dpi)


def plot_importance(by_group: pd.Series, by_cluster: pd.Series, prov: str, path: Path,
                    dpi: int) -> Path:
    """Grouped importance: registry groups and correlation clusters (mean |summed SHAP|)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, s, title in ((a1, by_group, "By feature group"),
                         (a2, by_cluster.head(15), "By correlation cluster (top 15)")):
        s = s.iloc[::-1]
        ax.barh(range(len(s)), s.values, color=SERIES[0], height=0.65, edgecolor=SURFACE,
                linewidth=2)
        ax.set_yticks(range(len(s)), s.index)
        for y, v in enumerate(s.values):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=7.5, color=TEXT_PRIMARY)
        ax.set_xlabel("mean |summed SHAP| (log-odds)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return finish(fig, "Grouped feature importance (correlated features summed)", prov, path,
                  dpi)


def plot_dependence(shap_df: pd.DataFrame, X: pd.DataFrame, feats: list[str], prov: str,
                    path: Path, dpi: int) -> Path:
    """SHAP vs feature value for the top numeric features."""
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for ax, f in zip(axes[0], feats, strict=True):
        x = pd.to_numeric(X[f], errors="coerce")
        ok = x.notna()
        ax.scatter(x[ok], shap_df.loc[ok, f], s=3, alpha=0.35, color=SERIES[0], linewidths=0,
                   rasterized=True)
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)
        hi = x[ok].quantile(0.99)
        ax.set_xlim(x[ok].min(), hi if hi > x[ok].min() else None)
        ax.set_xlabel(f"{f} (x-axis capped at 99th pct)")
        ax.set_ylabel("SHAP (log-odds)")
        ax.set_title(f, loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return finish(fig, "Dependence: how each value maps to its contribution", prov, path, dpi)


def plot_waterfall(contrib: pd.Series, base: float, raw: float, prob: float, label: str,
                   top_n: int, prov: str, path: Path, dpi: int) -> Path:
    """One customer's raw score built up from the base value, largest contributions first."""
    order = contrib.abs().sort_values(ascending=False).index
    shown = contrib[order[:top_n]]
    rest = contrib[order[top_n:]].sum()
    items = [*shown.items(), (f"{len(order) - top_n} other features", rest)]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(items) + 2.0))
    running = base
    for i, (_name, v) in enumerate(items):
        y = len(items) - 1 - i
        ax.barh(y, v, left=running, color=RAISE if v > 0 else LOWER, height=0.62,
                edgecolor=SURFACE, linewidth=1)
        ax.text(running + v + (0.02 if v >= 0 else -0.02), y, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=TEXT_PRIMARY)
        running += v
    ax.set_yticks(range(len(items)), [n for n, _ in reversed(items)])
    ax.axvline(base, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.axvline(raw, color=TEXT_PRIMARY, linewidth=1)
    ax.set_xlabel(f"raw score (log-odds). Dashed = base {base:.2f}; solid = this customer "
                  f"{raw:.2f}. Orange raises risk, blue lowers it.")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return finish(fig, f"Why this customer's score ({label}): calibrated churn probability "
                       f"{prob:.1%}", prov, path, dpi)


# ---------------------------------------------------------------- report


def render_section(data_cfg: dict[str, Any], cfg: dict[str, Any], snap: str, n: int,
                   base: float, feat_imp: pd.Series, grp_imp: pd.Series, clu_imp: pd.Series,
                   clusters: pd.DataFrame, examples: list[dict[str, Any]],
                   drivers: pd.DataFrame, figs: dict[str, Path]) -> str:
    """Markdown for the `explain` section of results.md."""
    rel = {k: p.relative_to(resolve_path("reports", data_cfg)).as_posix() for k, p in figs.items()}
    multi = clusters.groupby("cluster")["feature"].agg(list)
    multi = multi[multi.map(len) > 1]
    parts = ["## Phase 7: explainability", "", run_banner(data_cfg), "",
             f"Exact TreeSHAP for the Phase 6 LightGBM on snapshot **{snap}** "
             f"({fmt_int(n)} users). Values are contributions to the **raw model score** "
             f"(log-odds; base value {base:.3f} ≈ {sigmoid(base):.1%} raw). Sigmoid "
             "calibration is monotone, so directions carry over to the calibrated "
             "probability, but sizes are in log-odds. Additivity (base + sum = raw score) "
             "is checked for every user. SHAP describes what the **model** relies on; it is "
             "not evidence that any feature causes churn.", "",
             "### Global importance", "", f"![summary]({rel['summary']})", "",
             f"![importance]({rel['importance']})", ""]
    parts.append(md_table(["Feature group", "Mean |summed SHAP|", "Share"],
                          [[g, f"{v:.3f}", f"{v / grp_imp.sum():.1%}"]
                           for g, v in grp_imp.items()], "lrr"))
    parts += ["", f"Correlation clusters (|Spearman ρ| > {cfg['cluster_abs_rho']}) with more "
                  "than one member:", ""]
    parts.append(md_table(["Cluster", "Members"],
                          [[c, ", ".join(f"`{m}`" for m in ms)] for c, ms in multi.items()],
                          "ll"))
    parts += ["", "Top 15 single features (mean |SHAP|; correlated features split credit):",
              ""]
    parts.append(md_table(["Feature", "Mean |SHAP|"],
                          [[f"`{f}`", f"{v:.3f}"] for f, v in feat_imp.head(15).items()], "lr"))
    parts += ["", f"![dependence]({rel['dependence']})", "",
              "### Per-customer drivers and reason codes", "",
              f"Each customer gets up to {cfg['top_drivers']} risk-raising drivers (correlation "
              f"clusters with summed SHAP ≥ {cfg['min_driver_contribution']}), described by "
              "the cluster's strongest member for that customer. Stored for every user in "
              f"`{snap}_drivers.parquet`.", ""]
    per_user = drivers.groupby("msno").size()
    parts.append(f"Users with at least one driver: {fmt_int(per_user.size)} of {fmt_int(n)}; "
                 f"median drivers per user {per_user.median():.0f}.")
    top_first = drivers[drivers["rank"] == 1]["cluster"].value_counts().head(10)
    parts += ["", "Most common #1 driver:", ""]
    parts.append(md_table(["Cluster", "Users", "Share of users with a driver"],
                          [[c, fmt_int(v), f"{v / per_user.size:.1%}"]
                           for c, v in top_first.items()], "lrr"))
    for ex in examples:
        parts += ["", f"**Example: {ex['label']}** (calibrated churn probability "
                      f"{ex['prob']:.1%}, actual outcome {'churned' if ex['y'] else 'retained'})",
                  "", f"![{ex['label']}]({rel[ex['key']]})", ""]
        if ex["drivers"].empty:
            parts.append("No risk-raising drivers above the reporting floor.")
        else:
            parts += [f"{r.rank}. {r.reason} (contribution {r.contribution:+.2f})"
                      for r in ex["drivers"].itertuples()]
    return "\n".join(parts)


# ---------------------------------------------------------------- entry point


def main() -> int:
    """Explain the Phase 6 model on the configured snapshot(s)."""
    data_cfg, cfg = load_config("data"), load_config("explain")
    feat_cfg, snap_cfg, model_cfg = (load_config("features"), load_config("snapshots"),
                                     load_config("model"))
    templates = load_config("reason_codes")["templates"]
    specs = {s.name: s for s in all_specs(feat_cfg)}
    bundle = joblib.load(resolve_path(model_cfg["outputs"]["model_dir"], data_cfg)
                         / "churn_model.joblib")
    num, cat = bundle["numeric"], bundle["categorical"]
    out = cfg["outputs"]
    exp_dir, tab_dir, fig_dir = (resolve_path(out["explain_dir"], data_cfg),
                                 resolve_path(out["tables_dir"], data_cfg),
                                 resolve_path(out["figures_dir"], data_cfg))
    for d in (exp_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)
    dpi, seed = int(out["dpi"]), int(cfg["seed"])
    feat_dir = resolve_path(feat_cfg["outputs"]["features_dir"], data_cfg)
    snap_dir = resolve_path(snap_cfg["outputs"]["snapshot_dir"], data_cfg)
    pct_s = 100 if data_cfg["mode"] == "full" else data_cfg["user_sample_pct"]
    apply_style()

    for snap in cfg["explain_snapshots"]:
        prov = (f"Source: KKBox, mode {data_cfg['mode']}, {pct_s}% user sample (seed "
                f"{data_cfg['sample_seed']}); model trained on {bundle['train_snapshot']}, "
                f"explained on {snap}. SHAP shows model reliance, not causes.")
        ds = data.load_dataset(feat_dir / f"{snap}.parquet",
                               snap_dir / snap / "labels.parquet", num, cat)
        print(f"explaining {snap}: {len(ds.y):,} users", flush=True)
        ex = explain(bundle["pipeline"], ds.X, num, cat)
        prob = bundle["calibrator"].predict(1 / (1 + np.exp(-ex.raw_score)))

        cluster_of = groups.correlation_clusters(ds.X, num, cat, cfg["cluster_abs_rho"],
                                                 cfg["cluster_sample"], seed)
        feat_imp = groups.importance(ex.shap)
        names = groups.name_clusters(cluster_of, feat_imp)
        group_of = {f: specs[f].group for f in ex.shap.columns}
        grp_imp = groups.importance(groups.grouped(ex.shap, group_of))
        clu_imp = groups.importance(groups.grouped(ex.shap, cluster_of)).rename(index=names)
        clusters = pd.DataFrame({"feature": list(cluster_of),
                                 "cluster": [names[c] for c in cluster_of.values()]})

        drivers = reasons.top_drivers(ex.shap, ds.X, ds.msno, cluster_of, names, group_of,
                                      templates, int(cfg["top_drivers"]),
                                      float(cfg["min_driver_contribution"]))
        shap_out = ex.shap.astype("float32").assign(
            msno=ds.msno.to_numpy(), raw_score=ex.raw_score, p_churn=prob)
        shap_out[["msno", "raw_score", "p_churn", *ex.shap.columns]].to_parquet(
            exp_dir / f"{snap}_shap.parquet", index=False)
        drivers.to_parquet(exp_dir / f"{snap}_drivers.parquet", index=False)
        (exp_dir / f"{snap}_shap_meta.json").write_text(json.dumps(
            {"base_value": ex.base_value, "units": "log-odds of raw model score",
             "model_family": bundle["family"], "train_snapshot": bundle["train_snapshot"]},
            indent=2), encoding="utf-8")
        for name, frame in {"feature_importance": feat_imp.rename("mean_abs_shap"),
                            "group_importance": grp_imp.rename("mean_abs_shap"),
                            "cluster_importance": clu_imp.rename("mean_abs_shap")}.items():
            frame.to_csv(tab_dir / f"{snap}_{name}.csv", float_format="%.10g")
        clusters.to_csv(tab_dir / f"{snap}_clusters.csv", index=False)

        sample = ds.X.sample(n=min(cfg["global_sample"], len(ds.X)), random_state=seed).index
        top = list(feat_imp.index[:cfg["summary_top_n"]])
        dep = [f for f in feat_imp.index if specs[f].kind != "categorical"][
            :cfg["dependence_top_n"]]
        figs = {
            "summary": plot_summary(ex.shap.loc[sample], ds.X.loc[sample], top, prov,
                                    fig_dir / "phase7_shap_summary.png", dpi, seed),
            "importance": plot_importance(grp_imp, clu_imp, prov,
                                          fig_dir / "phase7_grouped_importance.png", dpi),
            "dependence": plot_dependence(ex.shap.loc[sample], ds.X.loc[sample], dep, prov,
                                          fig_dir / "phase7_dependence.png", dpi),
        }
        order = np.argsort(-prob, kind="stable")
        examples = []
        for k, e in enumerate(cfg["waterfall_examples"]):
            i = int(order[min(len(order) - 1, int(e["quantile"] * len(order)))])
            key = f"waterfall_{k}"
            figs[key] = plot_waterfall(ex.shap.iloc[i], ex.base_value, float(ex.raw_score[i]),
                                       float(prob[i]), e["label"], int(cfg["waterfall_top_n"]),
                                       prov, fig_dir / f"phase7_waterfall_{k}.png", dpi)
            examples.append({"label": e["label"], "key": key, "prob": float(prob[i]),
                             "y": int(ds.y[i]),
                             "drivers": drivers[drivers["msno"] == ds.msno.iat[i]]})
        write_section(resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg),
                      "explain", render_section(data_cfg, cfg, snap, len(ds.y), ex.base_value,
                                                feat_imp, grp_imp, clu_imp, clusters, examples,
                                                drivers, figs))
        print(f"phase 7: {snap} explained; {len(drivers):,} driver rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
