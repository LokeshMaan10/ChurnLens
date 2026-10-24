"""Phase 8 runner: Cox survival model, two CLTV methods, revenue backtest.

Run with `python -m churnlens.value.run` (or `make value`). Writes
data/processed/{mode}/value/{S}.parquet (per-user CLTV), reports/value/*, figures, and
the `value` section of results.md. BG/NBD + Gamma-Gamma is not used: see decision #50.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churnlens.analysis.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style, finish
from churnlens.config import load_config, resolve_path
from churnlens.reporting import fmt_int, md_table, run_banner, write_section
from churnlens.value import cltv, cox

BACKTEST_DECILES = 10


def ntd(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.0f}"


def realized_revenue(con: duckdb.DuckDBPyConnection, users: pd.Series, cutoff: Any,
                     days: int) -> pd.Series:
    """NTD paid (non-cancel) from the cutoff for `days` days. Evaluation only: post-cutoff
    data is read here to check CLTV, never to build it."""
    con.register("bt_users", pd.DataFrame({"msno": users.to_numpy()}))
    df = con.execute(f"""
        SELECT u.msno, COALESCE(SUM(t.actual_amount_paid), 0) AS revenue
        FROM bt_users u LEFT JOIN transactions t ON t.msno = u.msno AND t.is_cancel = 0
            AND t.transaction_date >= DATE '{cutoff}'
            AND t.transaction_date < DATE '{cutoff}' + INTERVAL {days} DAY
        GROUP BY u.msno""").df()
    con.unregister("bt_users")
    return df.set_index("msno")["revenue"].reindex(users.to_numpy()).fillna(0.0)


# ---------------------------------------------------------------- figures


def plot_hazards(summary: pd.DataFrame, prov: str, path: Path, dpi: int) -> Path:
    """Forest plot of hazard ratios with 95% CIs (log scale)."""
    s = summary.sort_values("exp(coef)")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.6))
    y = np.arange(len(s))
    ax.errorbar(s["exp(coef)"], y, xerr=[s["exp(coef)"] - s["exp(coef) lower 95%"],
                                         s["exp(coef) upper 95%"] - s["exp(coef)"]],
                fmt="o", color=SERIES[0], ecolor=TEXT_SECONDARY, markersize=5, capsize=2)
    ax.axvline(1, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, s.index)
    for yi, v in zip(y, s["exp(coef)"], strict=True):
        ax.text(v, yi + 0.25, f"{v:.2f}", ha="center", fontsize=7.5, color=TEXT_PRIMARY)
    ax.set_xlabel("hazard ratio (log scale; > 1 = churns sooner, association only)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return finish(fig, "Cox model: hazard ratios for time to churn", prov, path, dpi)


def plot_cltv(v: pd.DataFrame, prov: str, path: Path, dpi: int, seed: int) -> Path:
    """Formula vs survival CLTV (sampled scatter) and their distributions."""
    s = v.sample(n=min(20000, len(v)), random_state=seed)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.scatter(s["cltv_formula"], s["cltv_survival"], s=3, alpha=0.3, color=SERIES[0],
               linewidths=0, rasterized=True)
    hi = float(np.nanpercentile(s[["cltv_formula", "cltv_survival"]].to_numpy(), 99.5))
    a1.plot([0, hi], [0, hi], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    a1.set_xlim(0, hi)
    a1.set_ylim(0, hi)
    a1.set_xlabel("CLTV, formula method (NTD)")
    a1.set_ylabel("CLTV, survival method (NTD)")
    a1.set_title("Per-customer agreement", loc="left")
    bins = np.linspace(0, hi, 50)
    for colour, col, lab in ((SERIES[0], "cltv_formula", "formula"),
                             (SERIES[1], "cltv_survival", "survival")):
        a2.hist(v[col].clip(upper=hi), bins=bins, histtype="step", linewidth=2, color=colour,
                label=lab)
    a2.set_xlabel("CLTV (NTD, clipped at 99.5th pct)")
    a2.set_ylabel("users")
    a2.set_title("Distribution", loc="left")
    a2.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return finish(fig, "CLTV: formula vs survival method", prov, path, dpi)


def plot_backtest(dec: pd.DataFrame, prov: str, path: Path, dpi: int) -> Path:
    """Mean predicted vs realized revenue by predicted decile, both methods."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for colour, m in ((SERIES[0], "formula"), (SERIES[1], "survival")):
        d = dec[dec["method"] == m]
        ax.plot(d["decile"], d["predicted"], color=colour, marker="o", markersize=4,
                label=f"{m}: predicted")
        ax.plot(d["decile"], d["realized"], color=colour, marker="s", markersize=4,
                linestyle="--", label=f"{m}: realized")
    ax.set_xlabel("decile of predicted revenue (1 = lowest)")
    ax.set_ylabel("mean revenue per user (NTD)")
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return finish(fig, "Backtest: predicted vs realized revenue (renewing 30-day users)",
                  prov, path, dpi)


# ---------------------------------------------------------------- entry point


def main() -> int:
    """Fit Cox on the fit snapshot, value every scored snapshot, backtest."""
    data_cfg, cfg = load_config("data"), load_config("value")
    feat_cfg, snap_cfg, clean_cfg = (load_config("features"), load_config("snapshots"),
                                     load_config("cleaning"))
    out = cfg["outputs"]
    val_dir, tab_dir, fig_dir = (resolve_path(out["value_dir"], data_cfg),
                                 resolve_path(out["tables_dir"], data_cfg),
                                 resolve_path(out["figures_dir"], data_cfg))
    for d in (val_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)
    dpi, seed = int(out["dpi"]), int(cfg["cox"]["seed"])
    feat_dir = resolve_path(feat_cfg["outputs"]["features_dir"], data_cfg)
    snap_dir = resolve_path(snap_cfg["outputs"]["snapshot_dir"], data_cfg)
    pct_s = 100 if data_cfg["mode"] == "full" else data_cfg["user_sample_pct"]
    prov = (f"Source: KKBox, mode {data_cfg['mode']}, {pct_s}% user sample (seed "
            f"{data_cfg['sample_seed']}). Margin and discount rate are placeholder assumptions.")
    apply_style()
    con = duckdb.connect(str(resolve_path(clean_cfg["outputs"]["database"], data_cfg)),
                         read_only=True)
    fit_snap = cfg["fit_snapshot"]
    fit_cutoff = snap_cfg["snapshots"][fit_snap]["cutoff"]
    H, dpm = int(cfg["horizon_months"]), int(cfg["days_per_month"])
    d = cltv.monthly_discount(float(cfg["annual_discount_rate"]))

    print("fitting Cox model", flush=True)
    model, channels, fit_df = cox.fit(con, fit_cutoff, cfg["cox"])
    ph = cox.ph_test(model, fit_df, cfg["cox"]["ph_test_sample"], seed)
    summary = model.summary
    summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_csv(
        tab_dir / "cox_summary.csv", float_format="%.10g")
    ph.to_csv(tab_dir / "cox_ph_test.csv", index=False, float_format="%.10g")

    fit_feats = pd.read_parquet(feat_dir / f"{fit_snap}.parquet")
    fit_labels = pd.read_parquet(snap_dir / fit_snap / "labels.parquet")
    fit_all = fit_feats.merge(fit_labels, on="msno", validate="one_to_one")
    paid30 = (fit_all["last_amount_paid"] / fit_all["last_plan_days"].where(
        fit_all["last_plan_days"] > 0) * dpm)
    median_price = float(paid30[paid30 > 0].median())
    buckets = cfg["formula_segments"]["plan_buckets"]
    rates = cltv.segment_renewal_rates(cltv.segment_key(fit_all, buckets),
                                       fit_all["is_churn"].to_numpy(),
                                       int(cfg["formula_segments"]["min_users"]))
    rates.to_csv(tab_dir / "formula_segment_rates.csv", index=False, float_format="%.10g")
    rate_of = rates.set_index("segment")["renewal_rate"]
    overall_renewal = 1 - float(fit_all["is_churn"].mean())

    values: dict[str, pd.DataFrame] = {}
    for snap in cfg["score_snapshots"]:
        cutoff = snap_cfg["snapshots"][snap]["cutoff"]
        print(f"valuing {snap}", flush=True)
        f = pd.read_parquet(feat_dir / f"{snap}.parquet")
        price = cltv.monthly_price(f, median_price, dpm)
        margin_m = price * float(cfg["margin"])
        seg = cltv.segment_key(f, buckets)
        renewal = seg.map(rate_of).fillna(overall_renewal).to_numpy()
        c_formula = cltv.cltv_from_survival(margin_m.to_numpy(),
                                            cltv.formula_survival(renewal, H), d)
        spells = cox.current_spells(con, cutoff, f["msno"], cfg["cox"])
        has = f["msno"].isin(spells.index).to_numpy()
        sp = spells.loc[f.loc[has, "msno"]]
        X = cox.covariates(sp.reset_index(), channels)
        s = cox.conditional_survival(model, X, sp["duration"].astype(float).to_numpy(), H, dpm)
        c_surv = c_formula.copy()     # users without any spell fall back to the formula
        c_surv[has] = cltv.cltv_from_survival(margin_m.to_numpy()[has], s, d)
        s1 = np.full(len(f), np.nan)
        s1[has] = s[:, 1] if H > 1 else 1.0
        v = pd.DataFrame({"msno": f["msno"], "monthly_price": price.round(4),
                          "margin_month": margin_m.round(4), "formula_segment": seg,
                          "renewal_rate_formula": renewal.round(6),
                          "spell_age_days": np.where(has, f["msno"].map(
                              spells["duration"]).fillna(-1), np.nan),
                          "survival_1m": np.round(s1, 6), "has_spell": has,
                          "cltv_formula": c_formula.round(4), "cltv_survival": c_surv.round(4)})
        v.to_parquet(val_dir / f"{snap}.parquet", index=False)
        values[snap] = v

    # Backtest on the fit snapshot: renewers' realized revenue over the observed window.
    bt = cfg["backtest"]
    a = values[bt["snapshot"]].merge(fit_labels, on="msno").merge(
        fit_feats[["msno", "last_plan_days"]], on="msno")
    a["realized"] = realized_revenue(con, a["msno"], snap_cfg["snapshots"][bt["snapshot"]]
                                     ["cutoff"], int(bt["horizon_days"])).to_numpy()
    months = math.ceil(int(bt["horizon_days"]) / dpm)
    a["pred_formula"] = a["monthly_price"] * sum(a["renewal_rate_formula"] ** j
                                                 for j in range(months))
    a["pred_survival"] = a["monthly_price"] * np.where(
        a["has_spell"], 1 + a["survival_1m"].fillna(1) * (months > 1), 0) + np.where(
        a["has_spell"], 0, a["pred_formula"])
    renew = a[a["is_churn"] == 0]
    m30 = renew[renew["last_plan_days"] == dpm]
    back_rows, dec_rows = [], []
    for scope, frame in (("renewers, 30-day plans", m30), ("all renewers", renew)):
        for m in ("formula", "survival"):
            r = cltv.backtest(frame["realized"].to_numpy(), frame[f"pred_{m}"].to_numpy())
            back_rows.append({"scope": scope, "method": m, **r})
    for m in ("formula", "survival"):
        q = pd.qcut(m30[f"pred_{m}"].rank(method="first"), BACKTEST_DECILES,
                    labels=range(1, BACKTEST_DECILES + 1))
        g = m30.groupby(q, observed=True)
        dec_rows += [{"method": m, "decile": int(k), "predicted": float(x[f"pred_{m}"].mean()),
                      "realized": float(x["realized"].mean())} for k, x in g]
    back = pd.DataFrame(back_rows)
    dec = pd.DataFrame(dec_rows)
    back.to_csv(tab_dir / "backtest.csv", index=False, float_format="%.10g")
    con.close()

    figs = {"hazards": plot_hazards(summary, prov, fig_dir / "phase8_hazard_ratios.png", dpi),
            "cltv": plot_cltv(values["B"], prov, fig_dir / "phase8_cltv_methods.png", dpi, seed),
            "backtest": plot_backtest(dec, prov, fig_dir / "phase8_backtest.png", dpi)}
    rel = {k: p.relative_to(resolve_path("reports", data_cfg)).as_posix() for k, p in figs.items()}
    main_bt = back[back["scope"] == "renewers, 30-day plans"].set_index("method")
    best = main_bt["mae"].idxmin()
    (tab_dir / "cltv_choice.txt").write_text(best, encoding="utf-8")

    parts = ["## Phase 8: survival and CLTV", "", run_banner(data_cfg), "",
             f"**Cox model** on {fmt_int(len(fit_df))} subscription spells before the "
             f"{fit_snap} cutoff ({fmt_int(int(fit_df['event'].sum()))} churn events; 30-day "
             f"gap rule). Concordance {model.concordance_index_:.3f} (in-sample). Hazard ratios "
             "are associations, not causal effects.", "", f"![hazards]({rel['hazards']})", ""]
    parts.append(md_table(
        ["Covariate", "Hazard ratio", "95% CI", "p", "PH test p"],
        [[c, f"{r['exp(coef)']:.3f}", f"{r['exp(coef) lower 95%']:.3f}–"
          f"{r['exp(coef) upper 95%']:.3f}", f"{r['p']:.1e}",
          f"{ph.set_index('covariate').loc[c, 'p']:.1e}" if c in set(ph["covariate"]) else "—"]
         for c, r in summary.iterrows()], "lrlrr"))
    n_viol = int((ph["p"] < 0.05).sum())
    ph_n = fmt_int(min(len(fit_df), cfg["cox"]["ph_test_sample"]))
    parts += ["", f"Proportional-hazards check (Schoenfeld, {ph_n}-spell sample): {n_viol} of "
                  f"{len(ph)} covariates have p < 0.05, so their effect changes over spell "
                  "age. Hazard ratios are then averages over time; survival-based CLTV "
                  "inherits this approximation.", "",
              "### CLTV", "",
              f"CLTV = expected discounted margin over {H} months if the customer renews now. "
              f"Margin {cfg['margin']:.0%} and discount {cfg['annual_discount_rate']:.0%}/year "
              "are **placeholder assumptions**. Monthly price from the latest plan (free plans: "
              f"list price, else median {median_price:.0f} NTD). Formula: renewal rate of the "
              "user's segment (auto-renew x plan length) in snapshot A. Survival: Cox "
              "conditional survival given the current spell's age. **BG/NBD + Gamma-Gamma is "
              "not used**: it models non-contractual purchasing, whereas KKBox is a contract "
              "with observed expiry and churn, and prices are fixed plan amounts (decision #50).",
              "", f"![cltv]({rel['cltv']})", ""]
    rows = []
    for snap, v in values.items():
        for col in ("cltv_formula", "cltv_survival"):
            rows.append([snap, col.split("_")[1], ntd(v[col].mean()), ntd(v[col].median()),
                         ntd(v[col].quantile(0.9)), ntd(v[col].sum())])
        rows.append([snap, "rank agreement (Spearman)",
                     f"{v[['cltv_formula', 'cltv_survival']].corr('spearman').iat[0, 1]:.3f}",
                     "", "", ""])
    parts.append(md_table(["Snapshot", "Method", "Mean", "Median", "90th pct", "Total"], rows,
                          "llrrrr"))
    parts += ["", "Formula segments (snapshot A):", ""]
    parts.append(md_table(["Segment", "Users", "Churn rate", "Renewal rate used"],
                          [[r.segment, fmt_int(r.users), f"{r.churn_rate:.1%}",
                            f"{r.renewal_rate:.1%}" + (" (overall: segment too small)"
                                                       if r.uses_overall_rate else "")]
                           for r in rates.itertuples()], "lrrr"))
    parts += ["", f"### Backtest against realized revenue (snapshot {bt['snapshot']})", "",
              f"Revenue actually paid in the {bt['horizon_days']} days after the cutoff (the "
              "window the data covers) by users who renewed, vs each method's undiscounted "
              f"gross prediction for {months} months. Long plans pay lumpily, so the main scope "
              "is 30-day plans.", "", f"![backtest]({rel['backtest']})", ""]
    parts.append(md_table(["Scope", "Method", "Users", "Realized total", "Predicted total",
                           "Predicted / realized", "MAE per user", "Bias per user", "Spearman"],
                          [[r.scope, r.method, fmt_int(r.users), ntd(r.realized_total),
                            ntd(r.predicted_total), f"{r.total_ratio:.3f}", f"{r.mae:.1f}",
                            f"{r.bias:+.1f}", f"{r.spearman:.3f}"] for r in back.itertuples()],
                          "llrrrrrrr"))
    parts += ["", f"Lower backtest MAE on renewing 30-day users: **{best}** → used for value at "
                  "risk in Phase 9 (`reports/value/cltv_choice.txt`). The backtest covers only "
                  f"{months} months of a {H}-month CLTV, so it checks price and near-term "
                  "retention, not long-run lifetime."]
    write_section(resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg),
                  "value", "\n".join(parts))
    print(f"phase 8: CLTV for {list(values)}; backtest winner {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
