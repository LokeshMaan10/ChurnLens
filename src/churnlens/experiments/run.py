"""Phase 10 runner: power analysis, SIMULATED A/B test with holdout, SIMULATED uplift demo.

Run with `python -m churnlens.experiments.run` (or `make experiments`). Needs Phase 9
actions (to define the target population) and Phase 5 features. Every outcome analysed
here is SIMULATED with fake effects; only the power table uses no simulated data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churnlens.analysis.plotting import SERIES, TEXT_SECONDARY, apply_style, finish
from churnlens.config import load_config, resolve_path
from churnlens.experiments import simulate
from churnlens.experiments import stats as xs
from churnlens.features.registry import all_specs
from churnlens.modeling.data import model_columns
from churnlens.reporting import fmt_int, md_table, run_banner, write_section

S = simulate.SIMULATED


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def plot_uplift(dec: pd.DataFrame, prov: str, path: Path, dpi: int) -> Path:
    """Predicted vs true (simulated) uplift by predicted-uplift decile."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.38
    x = dec["decile"].to_numpy()
    ax.bar(x - w / 2, dec["true_uplift"] * 100, width=w, color=SERIES[0],
           label="true fake uplift (known in simulation)")
    ax.bar(x + w / 2, dec["predicted_uplift"] * 100, width=w, color=SERIES[1],
           label="T-learner predicted uplift")
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("decile of predicted uplift (1 = highest)")
    ax.set_ylabel("churn reduction (percentage points)")
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return finish(fig, f"{S}: T-learner predictions vs the fake heterogeneous effect", prov,
                  path, dpi)


def main() -> int:
    """Power table, simulated experiment analysis, simulated uplift demo, report."""
    data_cfg, cfg = load_config("data"), load_config("experiments")
    pri_cfg, feat_cfg = load_config("prioritize"), load_config("features")
    out = cfg["outputs"]
    tab_dir, fig_dir = resolve_path(out["tables_dir"], data_cfg), resolve_path(
        out["figures_dir"], data_cfg)
    tab_dir.mkdir(parents=True, exist_ok=True)
    alpha, power = float(cfg["alpha"]), float(cfg["power"])
    sim = cfg["simulation"]
    rng = np.random.default_rng(sim["seed"])

    # 1. Power / sample-size table (no simulation involved).
    ps = cfg["power_scenarios"]
    k = ps["treatment_share"] / (1 - ps["treatment_share"])
    power_rows = []
    for base in ps["baseline_rates"]:
        for mde in ps["relative_mdes"]:
            n_c, n_t = xs.sample_size(base, base * (1 - mde), alpha, power, k)
            power_rows.append({"control_churn": base, "relative_reduction": mde,
                               "absolute_reduction": base * mde, "n_control": n_c,
                               "n_treatment": n_t, "n_total": n_c + n_t})
    power_tab = pd.DataFrame(power_rows)

    # 2. SIMULATED experiment on Phase 9's high-risk users.
    snap = sim["snapshot"]
    actions = pd.read_parquet(resolve_path(pri_cfg["outputs"]["prioritize_dir"], data_cfg)
                              / f"{snap}_actions.parquet")
    feats = pd.read_parquet(resolve_path(feat_cfg["outputs"]["features_dir"], data_cfg)
                            / f"{snap}.parquet")
    pop = actions[actions["risk_band"] == "high"].merge(feats, on="msno", validate="one_to_one")
    pop = pop.sort_values("msno").reset_index(drop=True)      # fixed order before the RNG
    n = len(pop)
    treated = simulate.assign(n, sim["treatment_share"], rng)
    # Users with no plan (NULL auto-renew) count as not on auto-renew.
    manual = (pop["last_is_auto_renew"].fillna(0).astype(int) != 1).to_numpy()
    effect = simulate.fake_effect(manual, sim)
    p = pop["p_churn"].to_numpy()
    y = simulate.simulate_outcomes(p, treated, effect, rng)
    true_uplift = p * effect

    p_c = float(p.mean())
    p_t = float((p * (1 - effect)).mean())
    n_t, n_c = int(treated.sum()), int((~treated).sum())
    planned = {"population": n, "n_control": n_c, "n_treatment": n_t,
               "expected_control_churn": p_c, "expected_treated_churn": p_t,
               "power_for_simulated_effect": xs.achieved_power(p_c, p_t, n_c, n_t, alpha)}
    k_sim = sim["treatment_share"] / (1 - sim["treatment_share"])
    need_c, need_t = xs.sample_size(p_c, p_t, alpha, power, k_sim)
    planned.update({"required_n_control": need_c, "required_n_treatment": need_t})

    def analyse(mask: np.ndarray, label: str) -> dict:
        r = xs.two_proportion_test(int(y[mask & ~treated].sum()), int((mask & ~treated).sum()),
                                   int(y[mask & treated].sum()), int((mask & treated).sum()),
                                   alpha)
        return {"subgroup": label, **r.as_dict(),
                "true_fake_effect_pp": float((p[mask] * effect[mask]).mean())}

    ab = pd.DataFrame([analyse(np.ones(n, bool), "all high-risk users"),
                       analyse(manual, "not on auto-renew"),
                       analyse(~manual, "on auto-renew")])
    srm_ok = xs.srm_check(n_c, n_t, sim["treatment_share"], cfg["srm_alpha"])
    control_idx = np.flatnonzero(~treated)
    dropped = rng.choice(control_idx, size=int(len(control_idx) * sim["srm_demo_drop_share"]),
                         replace=False)
    srm_bad = xs.srm_check(n_c - len(dropped), n_t, sim["treatment_share"], cfg["srm_alpha"])

    # 3. SIMULATED uplift demo (T-learner) on the same simulated data.
    up = cfg["uplift"]
    num, _ = model_columns(all_specs(feat_cfg))
    X = pop[num].astype(float)
    test = simulate.assign(n, up["test_share"], np.random.default_rng(sim["seed"] + 1))
    pred = simulate.t_learner(X, treated, y, test, up, sim["seed"])
    dec = simulate.uplift_deciles(pred, true_uplift[test], treated[test], y[test], up["deciles"])
    corr = float(np.corrcoef(pred, true_uplift[test])[0, 1])

    for name, frame in {"power_table": power_tab, "simulated_ab": ab,
                        "simulated_uplift_deciles": dec}.items():
        frame.to_csv(tab_dir / f"{name}.csv", index=False, float_format="%.10g")

    pct_s = 100 if data_cfg["mode"] == "full" else data_cfg["user_sample_pct"]
    prov = (f"{S} outcomes with fake effects on KKBox users (mode {data_cfg['mode']}, {pct_s}% "
            f"sample, snapshot {snap}). Not evidence of real effects.")
    apply_style()
    fig = plot_uplift(dec, prov, fig_dir / "phase10_simulated_uplift.png", int(out["dpi"]))
    rel = fig.relative_to(resolve_path("reports", data_cfg)).as_posix()

    fe = sim["fake_effect"]
    parts = ["## Phase 10: experiments", "", run_banner(data_cfg), "",
             "### Power and sample size (real calculation, no simulation)", "",
             f"Two-sided two-proportion z-test, α = {alpha}, power = {power:.0%}, "
             f"{ps['treatment_share']:.0%} treated. Users needed to detect a relative churn "
             "reduction:", ""]
    parts.append(md_table(["Control churn", "Relative reduction", "Absolute (pp)",
                           "n control", "n treatment", "n total"],
                          [[pct(r.control_churn, 0), pct(r.relative_reduction, 0),
                            f"{r.absolute_reduction * 100:.2f}", fmt_int(r.n_control),
                            fmt_int(r.n_treatment), fmt_int(r.n_total)]
                           for r in power_tab.itertuples()], "rrrrrr"))
    parts += ["", f"### Planned retention test with holdout (design is real; outcomes below "
                  f"are {S})", "",
              f"Population: snapshot {snap} users in Phase 9's high-risk band "
              f"({fmt_int(n)} users), randomized {sim['treatment_share']:.0%} treatment / "
              f"{1 - sim['treatment_share']:.0%} holdout (seed {sim['seed']}). Expected control "
              f"churn (mean calibrated P) {pct(p_c)}.", "",
              f"- To detect the simulated average effect ({pct(p_c)} → {pct(p_t)}) at "
              f"{power:.0%} power needs {fmt_int(need_c)} control + {fmt_int(need_t)} treated; "
              f"the design has {fmt_int(n_c)} + {fmt_int(n_t)}, giving power "
              f"{planned['power_for_simulated_effect']:.0%}.", "",
              f"### {S} A/B result", "",
              f"**{S}.** Fake effects: churn reduced by {pct(fe['base_relative_reduction'], 0)} "
              f"for auto-renew users and {pct(fe['manual_renew_relative_reduction'], 0)} for "
              "others. Outcomes drawn from the model's calibrated probabilities.", ""]
    parts.append(md_table(
        ["Subgroup", "Control n", "Treated n", "Control churn", "Treated churn",
         "Difference (pp)", "95% CI (pp)", "p", "True fake effect (pp)"],
        [[r.subgroup, fmt_int(r.n_control), fmt_int(r.n_treatment), pct(r.rate_control),
          pct(r.rate_treatment), f"{r.difference * 100:+.2f}",
          f"{r.ci_low * 100:+.2f} to {r.ci_high * 100:+.2f}", f"{r.p_value:.1e}",
          f"{-r.true_fake_effect_pp * 100:+.2f}"] for r in ab.itertuples()], "lrrrrrlrr"))
    parts += ["", f"### Sample-ratio mismatch check ({S} assignment)", ""]
    parts.append(md_table(["Run", "Treated share observed", "Planned", "χ²", "p", "Alarm"],
                          [[lab, pct(r["observed_treatment_share"], 2),
                            pct(r["expected_treatment_share"], 0), f"{r['chi2']:.2f}",
                            f"{r['p_value']:.1e}", "**yes**" if r["mismatch"] else "no"]
                           for lab, r in (("as randomized", srm_ok),
                                          (f"{S} bug: {sim['srm_demo_drop_share']:.0%} of "
                                           "control lost", srm_bad))], "lrrrrl"))
    parts += ["", f"An SRM alarm (p < {cfg['srm_alpha']}) means the result should not be "
                  "trusted until the assignment or logging fault is found. The check's "
                  "sensitivity grows with sample size: with this ~1.6k-user holdout, a 5% loss "
                  "of control users was not detectable (p = 0.075 in a trial run).", "",
              f"### {S} uplift demo (T-learner)", "",
              f"**{S}.** Separate LightGBM churn models for treated and control on half the "
              "simulated population; predicted uplift = P(churn | control) − P(churn | treated) "
              f"on the other half. Correlation of predicted with the true fake uplift: "
              f"{corr:.3f}.", "", f"![uplift]({rel})", ""]
    parts.append(md_table(["Decile", "Users", "Predicted uplift (pp)", "True fake uplift (pp)",
                           "Observed simulated uplift (pp)"],
                          [[r.decile, fmt_int(r.users), f"{r.predicted_uplift * 100:.2f}",
                            f"{r.true_uplift * 100:.2f}", f"{r.observed_uplift * 100:.2f}"]
                           for r in dec.itertuples()], "rrrrr"))
    parts += ["", "Reading: the model's ranking points the right way (true fake uplift falls "
                  "from the top decile to the bottom), but its predicted magnitudes are far wider "
                  "than the truth: with a small control arm, T-learner differences are noisy. "
                  f"Everything in this section except the power table is {S}. Real save rates "
                  "and uplift require running the designed test."]
    write_section(resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg),
                  "experiments", "\n".join(parts))
    print(f"phase 10: power table {len(power_tab)} rows; {S} test on {n:,} users", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
