"""Phase 6 runner: tune, calibrate, select and threshold on A; evaluate once on B.

Run with `python -m churnlens.modeling.run` (or `make model`). Every choice (hyper-
parameters, calibration method, model family, threshold) is made on snapshot A; snapshot
B is only scored at the end. Logs to MLflow (local `mlruns/`).
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from churnlens.analysis.plotting import (
    SEQUENTIAL,
    SERIES,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    apply_style,
    finish,
)
from churnlens.config import PROJECT_ROOT, load_config, resolve_path
from churnlens.features.registry import all_specs
from churnlens.modeling import data, evaluate, threshold, train
from churnlens.reporting import fmt_int, md_table, run_banner, write_section

FAMILY_LABEL = {"logreg": "Logistic regression", "random_forest": "Random forest",
                "lightgbm": "LightGBM"}
HEATMAP_WHITE_TEXT_ABOVE = 0.45


def pct(x: float, d: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def f3(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


# ---------------------------------------------------------------- per-family training


def run_family(name: str, A: data.Dataset, B: data.Dataset, num: list[str], cat: list[str],
               cfg: dict[str, Any]) -> dict[str, Any]:
    """Tune, OOF, calibrate on A; fit on all of A; score B. Returns everything needed."""
    t0 = time.time()
    est = data.make_estimator(name, num, cat, cfg)
    tuned = train.tune(est, A.X, A.y, cfg["models"][name], cfg)
    est = est.set_params(**tuned.best_params)
    oof_raw = train.oof_predictions(est, A.X, A.y, cfg)
    method, cal_table, oof_cal = train.choose_calibration(oof_raw, A.y, cfg)
    calibrator = train.Calibrator(method).fit(oof_raw, A.y)
    final = train.fit_final(est, {}, A.X, A.y)
    b_raw = final.predict_proba(B.X)[:, 1]
    b_cal = calibrator.predict(b_raw)
    print(f"  {name}: CV PR-AUC {tuned.best_score:.4f}, calibration {method}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return {"name": name, "tuned": tuned, "pipeline": final, "calibrator": calibrator,
            "calibration": method, "calibration_table": cal_table,
            "oof_raw": oof_raw, "oof_cal": oof_cal, "b_raw": b_raw, "b_cal": b_cal,
            "oof_metrics": evaluate.metrics(A.y, oof_raw, oof_cal),
            "test_metrics": evaluate.metrics(B.y, b_raw, b_cal),
            "test_metrics_raw": evaluate.metrics(B.y, b_raw, b_raw),
            "seconds": time.time() - t0}


# ---------------------------------------------------------------- figures


def plot_curves(results: dict[str, dict], y: np.ndarray, prov: str, path: Path,
                dpi: int) -> Path:
    """PR and ROC curves on the test snapshot (two charts, one per metric)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for colour, (name, r) in zip(SERIES, results.items(), strict=False):
        pr, rc, _ = precision_recall_curve(y, r["b_raw"])
        a1.plot(rc, pr, color=colour,
                label=f"{FAMILY_LABEL[name]} (PR-AUC {r['test_metrics']['pr_auc']:.3f})")
        fpr, tpr, _ = roc_curve(y, r["b_raw"])
        a2.plot(fpr, tpr, color=colour,
                label=f"{FAMILY_LABEL[name]} (ROC-AUC {r['test_metrics']['roc_auc']:.3f})")
    a1.axhline(y.mean(), color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    a1.text(0.98, y.mean() + 0.01, f"base rate {y.mean():.1%}", ha="right", fontsize=8,
            color=TEXT_SECONDARY)
    a2.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    for ax, (xl, yl, t) in zip((a1, a2), (("recall", "precision", "Precision-recall"),
                                          ("false positive rate", "true positive rate",
                                           "ROC")), strict=True):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(t, loc="left")
        ax.legend(loc="upper right" if t != "ROC" else "lower right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return finish(fig, "Model comparison on the out-of-time test snapshot B", prov, path, dpi)


def plot_calibration(raw: pd.DataFrame, cal: pd.DataFrame, method: str, prov: str, path: Path,
                     dpi: int) -> Path:
    """Reliability diagram on B: raw (class-weighted) vs calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    top = max(raw["mean_predicted"].max(), cal["mean_predicted"].max(),
              cal["observed_rate"].max()) * 1.05
    ax.plot([0, top], [0, top], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.plot(raw["mean_predicted"], raw["observed_rate"], color=SERIES[1], marker="o",
            markersize=5, label="raw, class-weighted")
    ax.plot(cal["mean_predicted"], cal["observed_rate"], color=SERIES[0], marker="o",
            markersize=5, label=f"calibrated ({method})")
    ax.set_xlabel("mean predicted churn probability (decile)")
    ax.set_ylabel("observed churn rate")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return finish(fig, "Calibration on snapshot B (dashed = perfect)", prov, path, dpi)


def plot_lift_gains(lift: pd.DataFrame, prov: str, path: Path, dpi: int) -> Path:
    """Lift by risk decile and cumulative gains on B."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.bar(lift["bin"], lift["lift"], color=SERIES[0], width=0.7, edgecolor="#fcfcfb",
           linewidth=2)
    for b, v in zip(lift["bin"], lift["lift"], strict=True):
        a1.text(b, v + 0.1, f"{v:.1f}x", ha="center", fontsize=8, color=TEXT_PRIMARY)
    a1.axhline(1, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    a1.set_xticks(lift["bin"])
    a1.set_xlabel("risk decile (1 = highest predicted risk)")
    a1.set_ylabel("lift vs base rate")
    a1.set_title("Lift by decile", loc="left")
    x = np.concatenate([[0], lift["cumulative_population"]]) * 100
    g = np.concatenate([[0], lift["cumulative_gain"]]) * 100
    a2.plot(x, g, color=SERIES[0], marker="o", markersize=4, label="model")
    a2.plot([0, 100], [0, 100], color=TEXT_SECONDARY, linestyle="--", linewidth=1,
            label="random")
    a2.set_xlabel("% of users contacted (highest risk first)")
    a2.set_ylabel("% of churners reached")
    a2.set_title("Cumulative gains", loc="left")
    a2.legend(loc="lower right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return finish(fig, "Ranking quality on snapshot B", prov, path, dpi)


def plot_confusion(c: dict[str, Any], prov: str, path: Path, dpi: int) -> Path:
    """Confusion matrix at the chosen threshold (counts and row shares)."""
    m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
    share = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{int(v):,}\n{share[i, j]:.1%} of row", ha="center", va="center",
                fontsize=9, color="white" if share[i, j] > HEATMAP_WHITE_TEXT_ABOVE
                else TEXT_PRIMARY)
    ax.set_xticks([0, 1], ["not contacted", "contacted"])
    ax.set_yticks([0, 1], ["retained", "churned"])
    ax.set_xlabel("model decision")
    ax.set_ylabel("actual outcome")
    ax.grid(False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return finish(fig, f"Confusion matrix on B at threshold {c['threshold']:.3f}", prov,
                  path, dpi)


def plot_cost(curve_a: pd.DataFrame, curve_b: pd.DataFrame, t: float, prov: str, path: Path,
              dpi: int) -> Path:
    """Cost per user vs threshold on A (where it is chosen) and B (where it is checked)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(curve_a["threshold"], curve_a["cost_per_user"], color=SERIES[0],
            label="A (out-of-fold, used to choose)")
    ax.plot(curve_b["threshold"], curve_b["cost_per_user"], color=SERIES[1],
            label="B (test, not used to choose)")
    ax.axvline(t, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.text(t + 0.01, ax.get_ylim()[1] * 0.97, f"chosen {t:.3f}", fontsize=8, va="top",
            color=TEXT_PRIMARY)
    ax.set_xlabel("contact threshold on calibrated probability")
    ax.set_ylabel("cost per user (NTD, placeholder costs)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return finish(fig, "Expected cost by threshold", prov, path, dpi)


# ---------------------------------------------------------------- fairness


def fairness_groups(full: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.Series]:
    """Group labels for the audit: age band, gender, top cities, registration channel."""
    f = cfg["fairness"]
    edges = f["age_band_edges"]
    labels = [f"{lo}-{hi - 1}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    age = pd.cut(full["age"], bins=edges, right=False, labels=labels).astype(object) \
        .where(full["age"].notna(), "unknown")
    city = full["city"].fillna("unknown").astype(str)
    top = city[city != "unknown"].value_counts().index[:f["top_cities"]]
    city = city.where(city.isin(top) | (city == "unknown"), "other")
    return {"age band": age.astype(str), "gender": full["gender"].fillna("unknown"),
            "city": city.map(lambda c: c if c in ("unknown", "other") else f"city {c}"),
            "registration channel": full["registered_via"].fillna("unknown").map(
                lambda c: c if c == "unknown" else f"channel {c}")}


# ---------------------------------------------------------------- report


def render_section(data_cfg: dict[str, Any], cfg: dict[str, Any], costs: dict[str, Any],
                   results: dict[str, dict], best: str, t: float, ctx: dict[str, Any]) -> str:
    """Markdown for the `model` section of results.md."""
    rel = {k: p.relative_to(resolve_path("reports", data_cfg)).as_posix()
           for k, p in ctx["figures"].items()}
    r = results[best]
    parts = ["## Phase 6: modeling", "", run_banner(data_cfg), "",
             f"Trained, tuned, calibrated, selected and thresholded on snapshot "
             f"**{cfg['train_snapshot']}** ({fmt_int(ctx['n_train'])} users, churn "
             f"{pct(ctx['rate_train'], 2)}); evaluated once on out-of-time snapshot "
             f"**{cfg['test_snapshot']}** ({fmt_int(ctx['n_test'])} users, churn "
             f"{pct(ctx['rate_test'], 2)}). {ctx['n_features']} model features "
             "(demographics excluded). Class weights for imbalance; no SMOTE "
             "(decision #41). Tracked in MLflow experiment "
             f"`{cfg['mlflow']['experiment']}`, run `{ctx['run_id']}`.", "",
             "### Model comparison", "",
             "Selection uses snapshot A out-of-fold (OOF) PR-AUC only; B columns are reported "
             "after the choice was made. Ranking metrics (PR-AUC, ROC-AUC, top-K, lift, "
             "gains, curves) use the **raw model score**; probability metrics (log loss, "
             "Brier, calibration, mean prediction) and the contact threshold use the "
             "**calibrated probability** (decision #46).", ""]
    rows = []
    for name, x in results.items():
        o, b = x["oof_metrics"], x["test_metrics"]
        rows.append([FAMILY_LABEL[name] + (" **(selected)**" if name == best else ""),
                     f3(x["tuned"].best_score), f3(o["pr_auc"]), f3(o["roc_auc"]),
                     x["calibration"], f3(b["pr_auc"]), f3(b["roc_auc"]), f3(b["log_loss"]),
                     f"{b['brier']:.4f}", f"{x['seconds'] / 60:.1f} min"])
    parts.append(md_table(["Model", "A tuning CV PR-AUC", "A OOF PR-AUC", "A OOF ROC-AUC",
                           "Calibration", "B PR-AUC", "B ROC-AUC", "B log loss", "B Brier",
                           "Train time"], rows, "lrrrlrrrrr"))
    parts += ["", f"![curves]({rel['curves']})", "", "Best hyperparameters (A):", ""]
    parts.append(md_table(["Model", "Parameters"],
                          [[FAMILY_LABEL[n], "`" + json.dumps({k.replace("clf__", ""): (
                              round(v, 5) if isinstance(v, float) else v)
                              for k, v in x["tuned"].best_params.items()}) + "`"]
                           for n, x in results.items()], "ll"))

    parts += ["", f"### Calibration ({FAMILY_LABEL[best]})", "",
              "Class weights inflate raw probabilities; the calibration map is chosen by "
              "cross-validated Brier on A's OOF predictions and fitted on all of them. "
              f"Methods within {cfg['calibration']['brier_tie_tolerance']} Brier of the best "
              f"are tied; ties go to {' > '.join(cfg['calibration']['preference'])}. "
              f"Chosen: **{r['calibration']}**.", ""]
    parts.append(md_table(["Method (A OOF)", "Brier", "Log loss", "Mean predicted"],
                          [[m.method, f"{m.brier:.4f}", f"{m.log_loss:.4f}",
                            pct(m.mean_predicted, 2)]
                           for m in r["calibration_table"].itertuples()], "lrrr"))
    rb, cb = r["test_metrics_raw"], r["test_metrics"]
    parts += ["", f"On B: Brier {rb['brier']:.4f} raw → {cb['brier']:.4f} calibrated; mean "
                  f"prediction {pct(rb['mean_predicted'], 2)} raw → "
                  f"{pct(cb['mean_predicted'], 2)} calibrated vs actual "
                  f"{pct(cb['base_rate'], 2)}.", "", f"![calibration]({rel['calibration']})", ""]
    parts.append(md_table(["Decile", "Users", "Mean predicted", "Observed churn"],
                          [[c.bin, fmt_int(c.users), pct(c.mean_predicted, 2),
                            pct(c.observed_rate, 2)] for c in ctx["cal_table"].itertuples()],
                          "rrrr"))

    br = threshold.breakeven(costs)
    parts += ["", "### Threshold by expected cost", "",
              f"Costs from `configs/costs.yaml` (**placeholder assumptions**): offer "
              f"{costs['offer_cost']} {costs['currency']}, churn loss {costs['churn_loss']} "
              f"{costs['currency']}, save rate {pct(costs['save_rate'], 0)}. Analytic "
              f"break-even probability {br:.3f}; threshold chosen on A's calibrated OOF "
              f"predictions: **{t:.3f}**.", "", f"![cost]({rel['cost']})", ""]
    for snap, table in (("A (OOF)", ctx["policy_a"]), ("B (test)", ctx["policy_b"])):
        parts += [f"Policy cost, snapshot {snap}:", ""]
        parts.append(md_table(["Policy", "Contacted", "Share", "Churners contacted",
                               "Total cost", "Cost per user"],
                              [[p.policy, fmt_int(p.contacted), pct(p.contacted_share),
                                fmt_int(p.churners_contacted), f"{p.total_cost:,.0f}",
                                f"{p.cost_per_user:.2f}"] for p in table.itertuples()],
                              "lrrrrr"))
        parts.append("")

    c = ctx["confusion"]
    parts += ["### Test-snapshot results for the selected model", "",
              f"![confusion]({rel['confusion']})", "",
              f"At threshold {t:.3f}: precision {pct(c['precision'])}, recall "
              f"{pct(c['recall'])}, false positive rate {pct(c['fpr'])}, "
              f"{pct(c['contacted_share'])} of users contacted.", "",
              "Top-K (contact the highest-risk K% of B):", ""]
    parts.append(md_table(["Top K", "Contacted", "Churners caught", "Precision", "Recall",
                           "Lift"],
                          [[f"{k.top_pct}%", fmt_int(k.contacted), fmt_int(k.churners_caught),
                            pct(k.precision), pct(k.recall), f"{k.lift:.1f}x"]
                           for k in ctx["top_k"].itertuples()], "rrrrrr"))
    parts += ["", f"![lift]({rel['lift']})", ""]
    parts.append(md_table(["Decile", "Users", "Churn rate", "Lift", "Cumulative gain"],
                          [[d.bin, fmt_int(d.users), pct(d.churn_rate), f"{d.lift:.2f}x",
                            pct(d.cumulative_gain)] for d in ctx["lift"].itertuples()],
                          "rrrrr"))

    a, ab = r["oof_metrics"], r["test_metrics"]
    parts += ["", "### A vs B gap", "",
              f"PR-AUC {a['pr_auc']:.3f} on A (OOF) vs {ab['pr_auc']:.3f} on B; ROC-AUC "
              f"{a['roc_auc']:.3f} vs {ab['roc_auc']:.3f}. Known contributors from earlier "
              "phases: B's labels agree less with the rebuilt churn rule (Phase 2), and A's "
              "last week overlaps the Lunar New Year dip (Phase 3). These are candidate "
              "explanations, not tested ones."]

    parts += ["", "### Fairness audit (snapshot B, threshold applied)", "",
              "Demographics are not model inputs; error rates are still compared across "
              f"groups. Groups under {cfg['fairness']['min_group_size']} users are marked *small*."
              " FNR = churners the policy misses; FPR = retained users contacted.", ""]
    for attr, table in ctx["fairness"].items():
        parts += [f"**{attr.capitalize()}**", ""]
        parts.append(md_table(
            ["Group", "Users", "Churn rate", "Mean predicted", "Flagged", "Recall", "FNR", "FPR",
             "Precision"],
            [[f"{g.group}{' *small*' if g.small_group else ''}", fmt_int(g.users),
              pct(g.churn_rate), pct(g.mean_predicted), pct(g.flagged_share), pct(g.recall),
              pct(g.fnr), pct(g.fpr), pct(g.precision)] for g in table.itertuples()],
            "lrrrrrrrr"))
        parts.append("")

    if ctx.get("ablation"):
        ab_ = ctx["ablation"]
        parts += ["### Demographic ablation", "",
                  f"Same family and hyperparameters with age, gender and city added (not the "
                  f"deployed model): A OOF PR-AUC {ab_['oof_without']:.4f} → "
                  f"{ab_['oof_with']:.4f} ({ab_['oof_with'] - ab_['oof_without']:+.4f}); "
                  f"B PR-AUC {ab_['test_without']:.4f} → {ab_['test_with']:.4f} "
                  f"({ab_['test_with'] - ab_['test_without']:+.4f})."]
    return "\n".join(parts)


# ---------------------------------------------------------------- entry point


def setup_mlflow(m: dict[str, Any]) -> None:
    """Local SQLite tracking store + artifact folder, creating the experiment if needed."""
    db = PROJECT_ROOT / m["tracking_db"]
    artifacts = PROJECT_ROOT / m["artifacts_dir"]
    db.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    if mlflow.get_experiment_by_name(m["experiment"]) is None:
        mlflow.create_experiment(m["experiment"], artifact_location=artifacts.as_uri())
    mlflow.set_experiment(m["experiment"])


def main() -> int:
    """Train and evaluate every family, then report on the selected one."""
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    data_cfg, cfg, costs = load_config("data"), load_config("model"), load_config("costs")
    feat_cfg, snap_cfg = load_config("features"), load_config("snapshots")
    specs = all_specs(feat_cfg)
    num, cat = data.model_columns(specs)
    feat_dir = resolve_path(feat_cfg["outputs"]["features_dir"], data_cfg)
    snap_dir = resolve_path(snap_cfg["outputs"]["snapshot_dir"], data_cfg)
    out = cfg["outputs"]
    model_dir, pred_dir = (resolve_path(out["model_dir"], data_cfg),
                           resolve_path(out["predictions_dir"], data_cfg))
    tab_dir, fig_dir = resolve_path(out["tables_dir"], data_cfg), resolve_path(
        out["figures_dir"], data_cfg)
    for d in (model_dir, pred_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)
    tr, te = cfg["train_snapshot"], cfg["test_snapshot"]

    def load(s: str, n: list[str], c: list[str]) -> data.Dataset:
        return data.load_dataset(feat_dir / f"{s}.parquet", snap_dir / s / "labels.parquet",
                                 n, c)

    A, B = load(tr, num, cat), load(te, num, cat)
    pct_s = 100 if data_cfg["mode"] == "full" else data_cfg["user_sample_pct"]
    prov = (f"Source: KKBox, mode {data_cfg['mode']}, {pct_s}% user sample (seed "
            f"{data_cfg['sample_seed']}); trained on snapshot {tr}, tested on {te}.")
    apply_style()

    setup_mlflow(cfg["mlflow"])
    with mlflow.start_run(run_name="phase6") as parent:
        mlflow.log_params({"mode": data_cfg["mode"], "sample_pct": pct_s,
                           "train_snapshot": tr, "test_snapshot": te,
                           "n_features": len(num) + len(cat), "class_weight": cfg["class_weight"],
                           **{f"cost_{k}": v for k, v in costs.items() if k != "currency"}})
        results = {}
        for name in cfg["models"]:
            print(f"training {name}", flush=True)
            with mlflow.start_run(run_name=name, nested=True):
                r = run_family(name, A, B, num, cat, cfg)
                mlflow.log_params({k.replace("clf__", ""): v
                                   for k, v in r["tuned"].best_params.items()})
                mlflow.log_param("calibration", r["calibration"])
                mlflow.log_metrics({f"A_oof_{k}": v for k, v in r["oof_metrics"].items()})
                mlflow.log_metrics({f"B_{k}": v for k, v in r["test_metrics"].items()})
                mlflow.log_metric("A_tuning_cv_pr_auc", r["tuned"].best_score)
                results[name] = r
                r["tuned"].cv_table.to_csv(tab_dir / f"tuning_{name}.csv", index=False,
                                           float_format="%.10g")

        best = max(results, key=lambda n: results[n]["oof_metrics"]["pr_auc"])
        r = results[best]
        step = float(cfg["threshold_grid_step"])
        t = threshold.choose_threshold(A.y, r["oof_cal"], costs, step)
        ev = cfg["evaluation"]
        ctx: dict[str, Any] = {
            "n_train": len(A.y), "rate_train": float(A.y.mean()), "n_test": len(B.y),
            "rate_test": float(B.y.mean()), "n_features": len(num) + len(cat),
            "run_id": parent.info.run_id,
            "policy_a": threshold.policy_summary(A.y, r["oof_cal"], t, costs),
            "policy_b": threshold.policy_summary(B.y, r["b_cal"], t, costs),
            "confusion": evaluate.confusion(B.y, r["b_cal"], t),
            "top_k": evaluate.top_k(B.y, r["b_raw"], ev["top_k_pct"]),
            "lift": evaluate.lift_table(B.y, r["b_raw"], r["b_cal"], ev["lift_bins"]),
            "cal_table": evaluate.calibration_table(B.y, r["b_raw"], r["b_cal"],
                                                  ev["calibration_bins"]),
        }
        groups = fairness_groups(B.full, cfg)
        ctx["fairness"] = {attr: evaluate.fairness_table(g, B.y, r["b_cal"], t,
                                                         cfg["fairness"]["min_group_size"])
                           for attr, g in groups.items()}
        if cfg["demographic_ablation"]:
            num_d, cat_d = data.model_columns(specs, include_demographics=True)
            A_d, B_d = load(tr, num_d, cat_d), load(te, num_d, cat_d)
            est = data.make_estimator(best, num_d, cat_d, cfg).set_params(
                **r["tuned"].best_params)
            oof_d = train.oof_predictions(est, A_d.X, A_d.y, cfg)
            b_d = train.fit_final(est, {}, A_d.X, A_d.y).predict_proba(B_d.X)[:, 1]
            ctx["ablation"] = {
                "oof_without": evaluate.metrics(A.y, r["oof_raw"], r["oof_raw"])["pr_auc"],
                "oof_with": evaluate.metrics(A_d.y, oof_d, oof_d)["pr_auc"],
                "test_without": evaluate.metrics(B.y, r["b_raw"], r["b_raw"])["pr_auc"],
                "test_with": evaluate.metrics(B_d.y, b_d, b_d)["pr_auc"]}
            mlflow.log_metrics({f"ablation_{k}": v for k, v in ctx["ablation"].items()})

        raw_cal = evaluate.calibration_table(B.y, r["b_raw"], r["b_raw"], ev["calibration_bins"])
        dpi = int(out["dpi"])
        ctx["figures"] = {
            "curves": plot_curves(results, B.y, prov, fig_dir / "phase6_pr_roc.png", dpi),
            "calibration": plot_calibration(raw_cal, ctx["cal_table"], r["calibration"], prov,
                                            fig_dir / "phase6_calibration.png", dpi),
            "lift": plot_lift_gains(ctx["lift"], prov, fig_dir / "phase6_lift_gains.png", dpi),
            "confusion": plot_confusion(ctx["confusion"], prov,
                                        fig_dir / "phase6_confusion.png", dpi),
            "cost": plot_cost(threshold.cost_curve(A.y, r["oof_cal"], costs, step),
                              threshold.cost_curve(B.y, r["b_cal"], costs, step), t, prov,
                              fig_dir / "phase6_cost_threshold.png", dpi),
        }

        bundle = {"family": best, "pipeline": r["pipeline"], "calibrator": r["calibrator"],
                  "threshold": t, "numeric": num, "categorical": cat,
                  "train_snapshot": tr, "best_params": r["tuned"].best_params}
        model_path = model_dir / "churn_model.joblib"
        joblib.dump(bundle, model_path)
        for s, ds_, raw, calp in ((f"{tr}_oof", A, r["oof_raw"], r["oof_cal"]),
                                  (te, B, r["b_raw"], r["b_cal"])):
            pd.DataFrame({"msno": ds_.msno, "is_churn": ds_.y, "p_raw": raw,
                          "p_churn": calp}).round(10).to_parquet(pred_dir / f"{s}.parquet",
                                                                 index=False)
        tables = {"top_k": ctx["top_k"], "lift": ctx["lift"], "calibration_B": ctx["cal_table"],
                  "policy_A": ctx["policy_a"], "policy_B": ctx["policy_b"],
                  **{f"fairness_{k.replace(' ', '_')}": v for k, v in ctx["fairness"].items()}}
        for name, frame in tables.items():
            frame.to_csv(tab_dir / f"{name}.csv", index=False, float_format="%.10g")
        summary = {n: {"oof": x["oof_metrics"], "test": x["test_metrics"],
                       "calibration": x["calibration"],
                       "best_params": {k: (float(v) if isinstance(v, float) else v)
                                       for k, v in x["tuned"].best_params.items()}}
                   for n, x in results.items()}
        (tab_dir / "summary.json").write_text(json.dumps(
            {"selected": best, "threshold": t, "confusion": ctx["confusion"],
             "models": summary, "ablation": ctx.get("ablation")}, indent=2, default=float),
            encoding="utf-8")
        mlflow.log_param("selected_family", best)
        mlflow.log_param("threshold", t)
        mlflow.log_metrics({f"selected_B_{k}": v for k, v in r["test_metrics"].items()})
        mlflow.log_artifacts(str(tab_dir), artifact_path="tables")
        for p in ctx["figures"].values():
            mlflow.log_artifact(str(p), artifact_path="figures")
        mlflow.log_artifact(str(model_path), artifact_path="model")

    write_section(resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg),
                  "model", render_section(data_cfg, cfg, costs, results, best, t, ctx))
    print(f"phase 6: selected {best}, threshold {t:.3f}, B PR-AUC "
          f"{r['test_metrics']['pr_auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
