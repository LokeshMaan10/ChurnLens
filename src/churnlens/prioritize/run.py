"""Phase 9 runner: value at risk, quadrants, rule-based actions, budget optimisation.

Run with `python -m churnlens.prioritize.run` (or `make prioritize`). Needs Phase 6
predictions, Phase 7 drivers and Phase 8 CLTV. Writes
data/processed/{mode}/prioritize/{S}_actions.parquet, reports/prioritize/*, figures and
the `prioritize` section of results.md. Labels are read only to evaluate the selection.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churnlens.analysis.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style, finish
from churnlens.config import load_config, resolve_path
from churnlens.prioritize import engine
from churnlens.reporting import fmt_int, md_table, run_banner, write_section

QUADRANT_ORDER = ["high risk / high value", "high risk / low value", "low risk / high value",
                  "low risk / low value"]


def ntd(x: float) -> str:
    return f"{x:,.0f}"


def plot_quadrant(df: pd.DataFrame, t: float, v: float, prov: str, path: Path, dpi: int,
                  seed: int) -> Path:
    """Risk x value scatter (sample) with the band cut-offs."""
    s = df.sample(n=min(20000, len(df)), random_state=seed)
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    sel = s["selected"].to_numpy()
    ax.scatter(s.loc[~sel, "cltv"], s.loc[~sel, "p_churn"], s=3, alpha=0.25, color=SERIES[0],
               linewidths=0, rasterized=True, label="not selected")
    ax.scatter(s.loc[sel, "cltv"], s.loc[sel, "p_churn"], s=6, alpha=0.7, color=SERIES[1],
               linewidths=0, rasterized=True, label="selected by the optimizer")
    ax.axhline(t, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.axvline(v, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("CLTV (NTD, 12 months, placeholder margin)")
    ax.set_ylabel("calibrated P(churn) (log scale)")
    xmax = float(s["cltv"].quantile(0.995))
    ax.set_xlim(0, xmax)
    for (x, y, label) in ((xmax * 0.98, 0.9, "high risk / high value"),
                          (v * 0.02, 0.9, "high risk / low value")):
        ax.text(x, y, label, ha="right" if x > v else "left", fontsize=8, color=TEXT_PRIMARY)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return finish(fig, "Risk x value (dashed = risk threshold and median CLTV)", prov, path, dpi)


def main() -> int:
    """Score, assign actions, optimise the contact list, report."""
    data_cfg, cfg = load_config("data"), load_config("prioritize")
    model_cfg, value_cfg, explain_cfg = (load_config("model"), load_config("value"),
                                         load_config("explain"))
    out = cfg["outputs"]
    pri_dir, tab_dir, fig_dir = (resolve_path(out["prioritize_dir"], data_cfg),
                                 resolve_path(out["tables_dir"], data_cfg),
                                 resolve_path(out["figures_dir"], data_cfg))
    for d in (pri_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)
    snap = cfg["snapshot"]
    bundle = joblib.load(resolve_path(model_cfg["outputs"]["model_dir"], data_cfg)
                         / "churn_model.joblib")
    t = float(bundle["threshold"]) if cfg["risk_threshold"] == "from_model" \
        else float(cfg["risk_threshold"])
    method = cfg["cltv_method"]
    if method == "auto":
        method = (resolve_path(value_cfg["outputs"]["tables_dir"], data_cfg)
                  / "cltv_choice.txt").read_text(encoding="utf-8").strip()
    preds = pd.read_parquet(resolve_path(model_cfg["outputs"]["predictions_dir"], data_cfg)
                            / f"{snap}.parquet")
    value = pd.read_parquet(resolve_path(value_cfg["outputs"]["value_dir"], data_cfg)
                            / f"{snap}.parquet")
    drivers = pd.read_parquet(resolve_path(explain_cfg["outputs"]["explain_dir"], data_cfg)
                              / f"{snap}_drivers.parquet")
    top = drivers[drivers["rank"] == 1].set_index("msno")
    df = preds[["msno", "p_churn", "is_churn"]].merge(
        value[["msno", f"cltv_{method}"]], on="msno", validate="one_to_one").rename(
        columns={f"cltv_{method}": "cltv"})
    df["driver_feature"] = df["msno"].map(top["feature"])
    df["driver_group"] = df["msno"].map(top["group"])
    df["driver_cluster"] = df["msno"].map(top["cluster"])
    df["driver_reason"] = df["msno"].map(top["reason"])
    df["evar"] = df["p_churn"] * df["cltv"]
    value_cut = float(df["cltv"].quantile(cfg["value_quantile"]))
    df["risk_band"], df["value_band"], df["quadrant"] = engine.bands(
        df["p_churn"].to_numpy(), df["cltv"].to_numpy(), t, value_cut)

    actions = cfg["actions"]
    assigned = engine.assign_actions(df.drop(columns=["is_churn"]), cfg["rules"], actions)
    df["rule"], df["action"] = assigned["rule"], assigned["action"]
    cost, benefit = engine.expected_benefit(df["evar"].to_numpy(), df["action"], actions)
    df["action_cost"], df["expected_benefit"] = cost, benefit
    b = cfg["budget"]
    budget, cap = float(b["total_ntd"]), int(b["outreach_capacity"])
    lambdas = np.concatenate([[0.0], np.logspace(np.log10(b["lambda_min"]),
                                                 np.log10(b["lambda_max"]),
                                                 int(b["lambda_grid_points"]))])
    df["selected"] = engine.greedy_select(cost, benefit, budget, cap, lambdas)
    bound = engine.lp_upper_bound(cost, benefit, budget, cap)
    order = np.lexsort((df.index.to_numpy(), -df["expected_benefit"].to_numpy(),
                        ~df["selected"].to_numpy()))
    df["priority_rank"] = 0
    df.loc[df.index[order[:int(df['selected'].sum())]], "priority_rank"] = \
        np.arange(1, int(df["selected"].sum()) + 1)

    policies = {"optimizer (benefit − λ × cost)": df["selected"].to_numpy(),
                "highest P(churn) first": engine.baseline_select(
                    df["p_churn"].to_numpy(), cost, benefit, budget, cap),
                "highest EVaR first": engine.baseline_select(
                    df["evar"].to_numpy(), cost, benefit, budget, cap)}
    policies["optimizer, benefit per NTD only (for comparison)"] = engine.greedy_select(
        cost, benefit, budget, cap, np.array([b["lambda_max"]]))
    pol_rows = []
    for name, chosen in policies.items():
        pol_rows.append({"policy": name, "contacted": int(chosen.sum()),
                         "spend": float(cost[chosen].sum()),
                         "expected_saved_value": float((benefit + cost)[chosen].sum()),
                         "expected_net_benefit": float(benefit[chosen].sum()),
                         "actual_churners_contacted": int(df.loc[chosen, "is_churn"].sum()),
                         "churn_rate_contacted": float(df.loc[chosen, "is_churn"].mean())
                         if chosen.any() else np.nan})
    pol = pd.DataFrame(pol_rows)
    quad = df.groupby("quadrant").agg(users=("msno", "size"), mean_p=("p_churn", "mean"),
                                      mean_cltv=("cltv", "mean"), total_evar=("evar", "sum"),
                                      selected=("selected", "sum")).reindex(QUADRANT_ORDER)
    acts = df.groupby("action").agg(users=("msno", "size"), selected=("selected", "sum"),
                                    spend=("action_cost", lambda c: float(
                                        c[df.loc[c.index, "selected"]].sum())),
                                    mean_benefit=("expected_benefit", "mean")) \
        .sort_values("users", ascending=False)
    rule_counts = df["rule"].value_counts()

    keep = ["msno", "p_churn", "cltv", "evar", "risk_band", "value_band", "quadrant",
            "driver_cluster", "driver_feature", "driver_group", "driver_reason", "rule",
            "action", "action_cost", "expected_benefit", "selected", "priority_rank"]
    df[keep].to_parquet(pri_dir / f"{snap}_actions.parquet", index=False)
    for name, frame in {"policies": pol, "quadrants": quad.reset_index(),
                        "actions": acts.reset_index()}.items():
        frame.to_csv(tab_dir / f"{snap}_{name}.csv", index=False, float_format="%.10g")

    pct_s = 100 if data_cfg["mode"] == "full" else data_cfg["user_sample_pct"]
    prov = (f"Source: KKBox, mode {data_cfg['mode']}, {pct_s}% user sample (seed "
            f"{data_cfg['sample_seed']}), snapshot {snap}. Costs and save rates are placeholders.")
    apply_style()
    fig = plot_quadrant(df, t, value_cut, prov, fig_dir / "phase9_risk_value.png",
                        int(out["dpi"]), int(explain_cfg["seed"]))
    rel = fig.relative_to(resolve_path("reports", data_cfg)).as_posix()

    best = pol.iloc[0]
    parts = ["## Phase 9: prioritization and actions", "", run_banner(data_cfg), "",
             f"Snapshot **{snap}** ({fmt_int(len(df))} users). EVaR = calibrated P(churn) x "
             f"CLTV ({method} method, Phase 8 backtest winner). Risk band high if P ≥ {t:.3f} "
             f"(Phase 6 threshold); value band high if CLTV ≥ {ntd(value_cut)} NTD (median). "
             "Action **costs and save rates are placeholder assumptions** "
             "(`configs/prioritize.yaml`); expected benefits inherit them. Labels are used only "
             "to evaluate the selected list, never to choose it.", "",
             f"![risk-value]({rel})", "", "### Risk x value quadrants", ""]
    parts.append(md_table(["Quadrant", "Users", "Mean P(churn)", "Mean CLTV", "Total EVaR",
                           "Share of EVaR", "Selected"],
                          [[q, fmt_int(int(r.users)), f"{r.mean_p:.1%}", ntd(r.mean_cltv),
                            ntd(r.total_evar), f"{r.total_evar / quad['total_evar'].sum():.1%}",
                            fmt_int(int(r.selected))] for q, r in quad.iterrows()],
                          "lrrrrrr"))
    parts += ["", "### Rules engine (first match wins)", ""]
    parts.append(md_table(["Precedence", "Rule", "Condition", "Action", "Users matched"],
                          [[i + 1, r["name"], f"`{r['when']}`", r["action"],
                            fmt_int(int(rule_counts.get(r["name"], 0)))]
                           for i, r in enumerate(cfg["rules"])], "rllll"))
    parts += ["", "### Actions", ""]
    parts.append(md_table(["Action", "Description", "Cost", "Save rate", "Users assigned",
                           "Selected", "Spend"],
                          [[a, actions[a]["description"], actions[a]["cost"],
                            f"{actions[a]['save_rate']:.0%}", fmt_int(int(r.users)),
                            fmt_int(int(r.selected)), ntd(r.spend)]
                           for a, r in acts.iterrows()], "llrrrrr"))
    parts += ["", f"### Budget optimizer (budget {ntd(budget)} NTD, capacity {fmt_int(cap)} "
                  "contacts)", "",
              "Users ranked by expected benefit − λ × cost, with λ searched on a grid and the "
              "best feasible list kept (decision #53); compared with simpler rankings under "
              "the same limits, and with pure benefit-per-NTD ranking, which wastes contacts "
              "when the cap binds. Expected values use the placeholder save rates; the last two "
              "columns check the list against B's actual outcomes.", ""]
    parts.append(md_table(["Policy", "Contacted", "Spend", "Expected saved value",
                           "Expected net benefit", "Actual churners contacted",
                           "Churn rate of contacted"],
                          [[r.policy, fmt_int(r.contacted), ntd(r.spend),
                            ntd(r.expected_saved_value), ntd(r.expected_net_benefit),
                            fmt_int(r.actual_churners_contacted),
                            f"{r.churn_rate_contacted:.1%}"] for r in pol.itertuples()],
                          "lrrrrrr"))
    parts += ["", f"Optimizer expected net benefit {ntd(best.expected_net_benefit)} NTD vs an "
                  f"upper bound of {ntd(bound)} NTD "
                  f"({best.expected_net_benefit / bound:.1%} of the bound)." if bound else ""]
    top10 = df[df["selected"]].sort_values("priority_rank").head(10)
    parts += ["", "Top 10 of the contact list:", ""]
    parts.append(md_table(["Rank", "P(churn)", "CLTV", "EVaR", "Action", "#1 driver"],
                          [[int(r.priority_rank), f"{r.p_churn:.1%}", ntd(r.cltv), ntd(r.evar),
                            r.action, r.driver_reason or "—"] for r in top10.itertuples()],
                          "rrrrll"))
    write_section(resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg),
                  "prioritize", "\n".join(parts))
    print(f"phase 9: {int(df['selected'].sum())} users selected; expected net "
          f"{best.expected_net_benefit:,.0f} NTD", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
