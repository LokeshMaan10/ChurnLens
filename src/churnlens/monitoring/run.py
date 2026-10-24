"""Phase 14 runner: drift of features and predictions (reference A vs current B),
retraining verdict, Evidently report, and the monitoring agent's incident note.

Run with `python -m churnlens.monitoring.run` (or `make monitor`; schedule that command
with cron / Windows Task Scheduler to run it after each new snapshot).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from churnlens.analysis.plotting import SERIES, TEXT_PRIMARY, apply_style, finish
from churnlens.config import load_config, resolve_path
from churnlens.features.registry import all_specs
from churnlens.monitoring import psi
from churnlens.reporting import md_table, run_banner, write_section

SHORT_WINDOW_DAYS = 14   # listening windows treated as short-window for calendar checks


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, specs: list, cfg: dict[str, Any]
                  ) -> pd.DataFrame:
    """PSI, band and missing-share change for every registry feature."""
    p = cfg["psi"]
    rows = []
    for s in specs:
        r, c = ref[s.name], cur[s.name]
        value = (psi.psi_categorical(r, c, p["epsilon"]) if s.kind == "categorical"
                 else psi.psi_numeric(r.astype(float), c.astype(float), p["bins"],
                                      p["epsilon"]))
        m_ref, m_cur = float(r.isna().mean()), float(c.isna().mean())
        med = ({} if s.kind == "categorical" else
               {"median_ref": float(r.astype(float).median()),
                "median_cur": float(c.astype(float).median())})
        rows.append({"feature": s.name, "group": s.group, "kind": s.kind, **med,
                     "in_model": s.use_in_model, "psi": round(value, 6),
                     "band": psi.band(value, p["moderate"], p["major"]),
                     "missing_ref": round(m_ref, 6), "missing_cur": round(m_cur, 6),
                     "missing_flag": bool(abs(m_cur - m_ref) > cfg["missing_share_change"])})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def retrain_verdict(cfg: dict[str, Any], pred_psi: float, feat: pd.DataFrame,
                    calib_gap: float | None, pr_drop: float | None) -> dict[str, Any]:
    """Apply the retraining policy; list every condition that fired."""
    pol = cfg["retrain_policy"]
    model_feats = feat[feat["in_model"]]
    share_major = float((model_feats["band"] == "major").mean())
    reasons = []
    if pred_psi > pol["prediction_psi_above"]:
        reasons.append(f"prediction PSI {pred_psi:.3f} > {pol['prediction_psi_above']}")
    if share_major > pol["share_of_model_features_major"]:
        reasons.append(f"{share_major:.1%} of model features in major drift > "
                       f"{pol['share_of_model_features_major']:.0%}")
    if calib_gap is not None and abs(calib_gap) > pol["calibration_gap_above"]:
        reasons.append(f"calibration gap {calib_gap:+.4f} (mean predicted - observed) exceeds "
                       f"±{pol['calibration_gap_above']}")
    if pr_drop is not None and pr_drop > pol["pr_auc_drop_above"]:
        reasons.append(f"PR-AUC dropped {pr_drop:.3f} > {pol['pr_auc_drop_above']}")
    return {"retrain": bool(reasons), "reasons": reasons, "share_model_features_major":
            share_major, "calibration_gap": calib_gap, "pr_auc_drop": pr_drop}


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str], cfg: dict[str, Any],
                     path: Path) -> dict[str, Any]:
    """Evidently data-drift preset on a seeded sample; returns its drifted-column summary."""
    from evidently import Report
    from evidently.presets import DataDriftPreset
    e = cfg["evidently"]
    r = ref[cols].sample(n=min(e["sample"], len(ref)), random_state=e["seed"])
    c = cur[cols].sample(n=min(e["sample"], len(cur)), random_state=e["seed"])
    snap = Report([DataDriftPreset()]).run(current_data=c, reference_data=r)
    path.parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(path))
    summary: dict[str, Any] = {"html": str(path), "columns": len(cols)}
    try:
        for m in snap.dict()["metrics"]:
            if "DriftedColumnsCount" in str(m.get("metric_name", m.get("metric_id", ""))):
                v = m["value"]
                summary.update({"drifted_columns": int(v["count"]),
                                "drifted_share": float(v["share"])})
                break
    except (KeyError, TypeError, ValueError):
        summary["note"] = "drift summary not parsed from the Evidently snapshot"
    return summary


def plot_psi(feat: pd.DataFrame, pred_psi: float, cfg: dict[str, Any], prov: str, path: Path,
             dpi: int) -> Path:
    """Top features by PSI, with the moderate / major thresholds marked."""
    top = feat.head(20).iloc[::-1]
    colours = {"major": SERIES[1], "moderate": SERIES[3], "stable": SERIES[0]}
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top["feature"], top["psi"], color=[colours[b] for b in top["band"]], height=0.65)
    for y, (v, b) in enumerate(zip(top["psi"], top["band"], strict=True)):
        ax.text(v, y, f" {v:.3f} ({b})", va="center", fontsize=7.5, color=TEXT_PRIMARY)
    for x, lab in ((cfg["psi"]["moderate"], "moderate"), (cfg["psi"]["major"], "major")):
        ax.axvline(x, color=TEXT_PRIMARY, linestyle="--", linewidth=0.8)
        ax.text(x, len(top) - 0.3, f" {lab}", fontsize=7.5, color=TEXT_PRIMARY)
    ax.set_xlabel(f"PSI (reference A vs current B); prediction PSI = {pred_psi:.3f}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return finish(fig, "Feature drift: top 20 by PSI", prov, path, dpi)


def main() -> int:
    """Compute drift, write the summary the drift_report tool serves, run the agent."""
    data_cfg, cfg = load_config("data"), load_config("monitoring")
    feat_cfg, model_cfg, snap_cfg = (load_config("features"), load_config("model"),
                                     load_config("snapshots"))
    out = cfg["outputs"]
    tab_dir = resolve_path(out["tables_dir"], data_cfg)
    tab_dir.mkdir(parents=True, exist_ok=True)
    ref_s, cur_s = cfg["reference_snapshot"], cfg["current_snapshot"]
    feat_dir = resolve_path(feat_cfg["outputs"]["features_dir"], data_cfg)
    ref = pd.read_parquet(feat_dir / f"{ref_s}.parquet")
    cur = pd.read_parquet(feat_dir / f"{cur_s}.parquet")
    specs = all_specs(feat_cfg)
    feat = feature_drift(ref, cur, specs, cfg)
    cutoffs = snap_cfg["snapshots"]
    gap_days = (cutoffs[cur_s]["cutoff"] - cutoffs[ref_s]["cutoff"]).days
    # Clock check on users present in both snapshots: a time-since feature whose per-user
    # value moved by ~the cutoff gap is tracking the calendar, not behaviour. (Comparing
    # population medians is misleading because the two snapshots hold different users.)
    ck = cfg["clock_features"]
    both = ref.merge(cur, on="msno", suffixes=("_ref", "_cur"))
    per_user = {}
    for f in feat.loc[feat["feature"].str.match(ck["name_pattern"]), "feature"]:
        per_user[f] = float((both[f"{f}_cur"] - both[f"{f}_ref"]).median())
    feat["per_user_median_change"] = feat["feature"].map(per_user)
    feat["clock_shift"] = (feat["per_user_median_change"] - gap_days).abs() \
        <= ck["tolerance_days"]

    pred_dir = resolve_path(model_cfg["outputs"]["predictions_dir"], data_cfg)
    p_ref = pd.read_parquet(pred_dir / f"{ref_s}_oof.parquet")
    p_cur = pd.read_parquet(pred_dir / f"{cur_s}.parquet")
    pp = cfg["psi"]
    pred_psi = psi.psi_numeric(p_ref["p_churn"], p_cur["p_churn"], pp["bins"], pp["epsilon"])
    calib_gap = float(p_cur["p_churn"].mean() - p_cur["is_churn"].mean())
    summary_model = json.loads((resolve_path(model_cfg["outputs"]["tables_dir"], data_cfg)
                                / "summary.json").read_text(encoding="utf-8"))
    sel = summary_model["models"][summary_model["selected"]]
    pr_drop = float(sel["oof"]["pr_auc"] - sel["test"]["pr_auc"])
    verdict = retrain_verdict(cfg, pred_psi, feat, calib_gap, pr_drop)

    cutoff = snap_cfg["snapshots"][ref_s]["cutoff"]
    window_start = cutoff - dt.timedelta(days=SHORT_WINDOW_DAYS)
    events = [e["name"] for e in cfg["calendar_events"]
              if e["start"] < cutoff and e["end"] >= window_start]

    num_model = [s.name for s in specs if s.use_in_model and s.kind != "categorical"]
    ev = {}
    if cfg["evidently"]["enabled"]:
        ev = evidently_report(ref.assign(p_churn=p_ref.set_index("msno").loc[
            ref["msno"], "p_churn"].to_numpy()), cur.assign(p_churn=p_cur.set_index("msno").loc[
            cur["msno"], "p_churn"].to_numpy()), [*num_model, "p_churn"], cfg,
            resolve_path(out["evidently_html"], data_cfg))

    summary = {
        "reference": ref_s, "current": cur_s,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "n_features": len(feat), "n_major": int((feat["band"] == "major").sum()),
        "n_moderate": int((feat["band"] == "moderate").sum()),
        "n_missing_flags": int(feat["missing_flag"].sum()),
        "prediction": {"psi": round(pred_psi, 6),
                       "band": psi.band(pred_psi, pp["moderate"], pp["major"]),
                       "reference_mean": float(p_ref["p_churn"].mean()),
                       "current_mean": float(p_cur["p_churn"].mean()),
                       "current_observed_churn": float(p_cur["is_churn"].mean())},
        "retrain": verdict, "calendar_events_in_reference_window": events, "evidently": ev,
        "cutoff_gap_days": gap_days,
        "features": feat.to_dict(orient="records")}
    summary_path = resolve_path(out["summary_json"], data_cfg)
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    feat.to_csv(tab_dir / "feature_psi.csv", index=False, float_format="%.10g")

    from churnlens.agents.specialists import monitoring as monitoring_agent
    agent = monitoring_agent("Scheduled drift check", write_note=True)

    pct_s = 100 if data_cfg["mode"] == "full" else data_cfg["user_sample_pct"]
    prov = (f"Source: KKBox, mode {data_cfg['mode']}, {pct_s}% user sample (seed "
            f"{data_cfg['sample_seed']}); reference {ref_s}, current {cur_s}.")
    apply_style()
    fig = plot_psi(feat, pred_psi, cfg, prov, resolve_path(out["figures_dir"], data_cfg)
                   / "phase14_psi.png", int(out["dpi"]))
    rel = fig.relative_to(resolve_path("reports", data_cfg)).as_posix()
    by_band = feat.groupby(["band", "in_model"]).size().unstack(fill_value=0)
    pol = cfg["retrain_policy"]
    parts = ["## Phase 14: monitoring", "", run_banner(data_cfg), "",
             f"Reference **{ref_s}** (training) vs current **{cur_s}**. Hand-written PSI "
             f"({pp['bins']} reference-quantile bins, missing values as their own bin; bands "
             f"stable < {pp['moderate']} ≤ moderate ≤ {pp['major']} < major) on all "
             f"{len(feat)} features and on the model's scores. Evidently data-drift report: "
             f"`{Path(ev.get('html', '')).name}`"
             + (f" ({ev['drifted_columns']} of {ev['columns']} columns flagged by Evidently's "
                f"own tests)" if "drifted_columns" in ev else "") + ".", "",
             f"![psi]({rel})", "", "### Summary", ""]
    parts.append(md_table(["Measure", "Value"], [
        ["Prediction PSI", f"{pred_psi:.3f} ({summary['prediction']['band']})"],
        ["Mean predicted churn: A out-of-fold → B",
         f"{summary['prediction']['reference_mean']:.2%} → "
         f"{summary['prediction']['current_mean']:.2%}"],
        ["Observed churn in B", f"{summary['prediction']['current_observed_churn']:.2%}"],
        ["Features in major / moderate drift", f"{summary['n_major']} / {summary['n_moderate']}"],
        ["Missing-share flags (> "
         f"{cfg['missing_share_change']:.0%} change)", str(summary["n_missing_flags"])],
        ["Calendar events overlapping A's last 14 days", ", ".join(events) or "none"],
    ], "lr"))
    parts += ["", "Drifted features by band (model features vs others):", ""]
    parts.append(md_table(["Band", *[("in model" if c else "not in model")
                                     for c in by_band.columns]],
                          [[b, *[int(v) for v in r]] for b, r in by_band.iterrows()],
                          "l" + "r" * len(by_band.columns)))
    parts += ["", "Top 10 features by PSI:", ""]
    parts.append(md_table(["Feature", "Group", "PSI", "Band", "Missing A → B"],
                          [[f"`{r.feature}`", r.group, f"{r.psi:.3f}", r.band,
                            f"{r.missing_ref:.1%} → {r.missing_cur:.1%}"]
                           for r in feat.head(10).itertuples()], "llrll"))
    parts += ["", "### Retraining policy (`configs/monitoring.yaml`)", "",
              f"Retrain if any holds: prediction PSI > {pol['prediction_psi_above']}; > "
              f"{pol['share_of_model_features_major']:.0%} of model features in major drift; "
              f"|calibration gap| > {pol['calibration_gap_above']}; PR-AUC drop > "
              f"{pol['pr_auc_drop_above']}. Verdict: **"
              f"{'RETRAIN' if verdict['retrain'] else 'no retrain'}**"
              + (" — " + "; ".join(verdict["reasons"]) if verdict["reasons"] else "") + ".", "",
              "### Monitoring agent incident note (offline mode)", "",
              f"Written to `{Path(out['incident_note']).as_posix()}`. Classification: "
              + "; ".join(f"{k}: {len(v)}" for k, v in agent.data.get("groups", {}).items())
              + "."]
    write_section(resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg),
                  "monitoring", "\n".join(parts))
    print(f"phase 14: prediction PSI {pred_psi:.3f}; {summary['n_major']} major features; "
          f"retrain={verdict['retrain']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
