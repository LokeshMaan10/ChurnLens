"""Phase 5 runner: build the feature table for every snapshot and document it.

Run with `python -m churnlens.features.build` (or `make features`). Needs Phase 2
snapshots and Phase 4 RFM outputs. Writes:
  data/processed/{mode}/features/{S}.parquet    msno + features (labels stay separate)
  reports/features/feature_registry.json        every feature with docs + use_in_model
  reports/features/univariate_{ref}.csv          per-feature association with churn (A only)
  the `features` section of results.md and of docs/data_dictionary.md
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from churnlens.analysis.audit import DICTIONARY_HEADER
from churnlens.analysis.stats import chi2_independence
from churnlens.config import load_config, resolve_path
from churnlens.features import behavior, members, transactions
from churnlens.features.registry import FeatureSpec, all_specs
from churnlens.reporting import fmt_int, md_table, run_banner, write_section

MISSING = "missing"
TOP_N_UNIVARIATE = 25
HIGH_MISSING_SHARE = 0.05


def build_snapshot(con: duckdb.DuckDBPyConnection, snap_dir: Path, rfm_file: Path,
                   cutoff: dt.date, cfg: dict[str, Any], specs: list[FeatureSpec]) -> pd.DataFrame:
    """Join every feature group for one snapshot; check it against the registry."""
    if not rfm_file.exists():
        raise FileNotFoundError(f"{rfm_file} missing: run `python -m churnlens.segmentation.run`")
    df = con.execute(f"""
        SELECT * FROM ({behavior.listening_sql(snap_dir, cutoff, cfg)}) b
        LEFT JOIN ({transactions.transactions_sql(snap_dir, cutoff, cfg)}) t USING (msno)
        LEFT JOIN ({members.members_sql(snap_dir, cutoff)}) m USING (msno)
        LEFT JOIN ({members.rfm_sql(rfm_file)}) r USING (msno)
        ORDER BY msno""").df()
    names = [s.name for s in specs]
    produced = [c for c in df.columns if c != "msno"]
    missing, extra = set(names) - set(produced), set(produced) - set(names)
    if missing or extra:
        raise AssertionError(f"registry mismatch: missing {sorted(missing)}, extra {sorted(extra)}")
    labels_path = (snap_dir / "labels.parquet").as_posix()
    n_labels = con.execute(f"SELECT COUNT(*) FROM read_parquet('{labels_path}')").fetchone()[0]
    if len(df) != n_labels or df["msno"].duplicated().any():
        raise AssertionError(f"expected one row per label user ({n_labels}), got {len(df)}")
    df = df[["msno", *names]]
    floats = df.select_dtypes("float").columns
    df[floats] = df[floats].round(int(cfg["float_decimals"]))
    return df


# ---------------------------------------------------------------- univariate summary


def univariate(features: pd.DataFrame, labels: pd.DataFrame,
               specs: list[FeatureSpec]) -> pd.DataFrame:
    """Per-feature coverage and association with churn (reference snapshot only).

    Numeric/binary: ROC-AUC of the raw value on rows where it is present (0.5 = none;
    below 0.5 means higher values go with lower churn). Categorical: Cramér's V, with
    NULL as its own category.
    """
    df = features.merge(labels, on="msno", validate="one_to_one")
    y = df["is_churn"].to_numpy()
    rows = []
    for s in specs:
        col = df[s.name]
        present = col.notna().to_numpy()
        row: dict[str, Any] = {"feature": s.name, "group": s.group, "kind": s.kind,
                               "use_in_model": s.use_in_model,
                               "coverage": float(present.mean()), "auc": np.nan,
                               "cramers_v": np.nan, "median_retained": np.nan,
                               "median_churned": np.nan}
        if s.kind == "categorical":
            g = col.fillna(MISSING).astype(str)
            row["cramers_v"] = chi2_independence(
                [(int(y[g == v].sum()), int((g == v).sum())) for v in g.unique()])[2]
        else:
            x = col.to_numpy(dtype=float)
            if present.sum() and len(np.unique(y[present])) == 2 and \
                    len(np.unique(x[present])) > 1:
                row["auc"] = float(roc_auc_score(y[present], x[present]))
            row["median_retained"] = float(np.nanmedian(x[(y == 0) & present])) \
                if ((y == 0) & present).any() else np.nan
            row["median_churned"] = float(np.nanmedian(x[(y == 1) & present])) \
                if ((y == 1) & present).any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out["strength"] = np.where(out["kind"] == "categorical", out["cramers_v"],
                               (out["auc"] - 0.5).abs() * 2)
    return out


# ---------------------------------------------------------------- docs + report


def render_dictionary(specs: list[FeatureSpec], data_cfg: dict[str, Any]) -> str:
    """The `features` section of docs/data_dictionary.md."""
    parts = ["## Model features", "",
             "_Generated from `src/churnlens/features/registry.py` by "
             "`python -m churnlens.features.build`._", "", run_banner(data_cfg), "",
             "Every feature uses only data strictly before its snapshot's cutoff. "
             "**Leakage risk** `low` = timestamped pre-cutoff events only; `medium` = "
             "pre-cutoff, but the value could have been restated later or comes from an "
             "undated extract (see note). **In model** = allowed as a model input.", ""]
    groups: dict[str, list[FeatureSpec]] = {}
    for s in specs:
        groups.setdefault(s.group, []).append(s)
    for group, items in groups.items():
        parts += [f"### {group[:1].upper()}{group[1:]}", ""]
        parts.append(md_table(
            ["Feature", "Type", "Definition", "Why it matters", "Leakage risk", "In model"],
            [[f"`{s.name}`", s.kind, s.definition, s.rationale,
              f"{s.leakage_risk}: {s.leakage_note}", "yes" if s.use_in_model else "**no**"]
             for s in items], "llllll"))
        parts.append("")
    return "\n".join(parts)


def pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{100 * x:.1f}%"


def render_section(data_cfg: dict[str, Any], cfg: dict[str, Any], ref: str,
                   specs: list[FeatureSpec], shapes: dict[str, tuple[int, int]],
                   uni: pd.DataFrame) -> str:
    """The `features` section of results.md."""
    in_model = [s for s in specs if s.use_in_model]
    groups = pd.Series([s.group for s in specs]).value_counts()
    parts = ["## Phase 5: feature engineering", "", run_banner(data_cfg), "",
             f"{len(specs)} features in {len(groups)} groups; {len(in_model)} allowed in the "
             f"model ({len(specs) - len(in_model)} demographic features kept out, built for the "
             "fairness audit). Definitions, rationale and leakage risk: "
             "`docs/data_dictionary.md` (Model features).", ""]
    parts.append(md_table(["Snapshot", "Users", "Feature columns"],
                          [[s, fmt_int(r), c] for s, (r, c) in shapes.items()], "lrr"))
    parts += ["", md_table(["Group", "Features"], [[g, n] for g, n in groups.items()], "lr"), "",
              f"### Strongest single-feature associations with churn (snapshot {ref})", "",
              "Numeric: ROC-AUC on rows where the feature is present (below 0.5 = higher "
              "values go with lower churn); strength = |AUC − 0.5| × 2. Categorical: "
              "Cramér's V. One feature at a time; correlated features overlap heavily, and "
              "none of this is causal.", ""]
    alarm = float(cfg["leakage_alarm_strength"])
    flagged = uni[uni["strength"] > alarm]
    strongest = uni.loc[uni["strength"].idxmax()]
    parts.append(
        f"**Leakage alarm** (any single feature with strength > {alarm}): "
        + ("none. " if flagged.empty else
           f"**{len(flagged)} flagged: {', '.join(f'`{f}`' for f in flagged['feature'])}.** ")
        + f"Strongest is `{strongest['feature']}` at {strongest['strength']:.3f}.")
    parts.append("")
    top = uni.sort_values("strength", ascending=False).head(TOP_N_UNIVARIATE)
    parts.append(md_table(
        ["Feature", "Group", "Coverage", "AUC", "Cramér's V", "Median retained",
         "Median churned", "In model"],
        [[f"`{r.feature}`", r.group, pct(r.coverage),
          "—" if pd.isna(r.auc) else f"{r.auc:.3f}",
          "—" if pd.isna(r.cramers_v) else f"{r.cramers_v:.3f}",
          "—" if pd.isna(r.median_retained) else f"{r.median_retained:g}",
          "—" if pd.isna(r.median_churned) else f"{r.median_churned:g}",
          "yes" if r.use_in_model else "no"] for r in top.itertuples()], "llrrrrrl"))
    high = uni[1 - uni["coverage"] > HIGH_MISSING_SHARE].sort_values("coverage")
    parts += ["", f"### Features missing for more than {HIGH_MISSING_SHARE:.0%} of users "
                  f"(snapshot {ref})", "",
              "Missing values are kept as NULL (meaningful: e.g. no plays means no skip rate); "
              "Phase 6 decides imputation per model.", ""]
    parts.append(md_table(["Feature", "Missing"], [[f"`{r.feature}`", pct(1 - r.coverage)]
                                                   for r in high.itertuples()], "lr"))
    return "\n".join(parts)


# ---------------------------------------------------------------- entry point


def main() -> int:
    """Build features for every snapshot, document, summarise."""
    data_cfg, cfg = load_config("data"), load_config("features")
    snap_cfg, rfm_cfg = load_config("snapshots"), load_config("rfm")
    ref = rfm_cfg["reference_snapshot"]
    specs = all_specs(cfg)
    base = resolve_path(snap_cfg["outputs"]["snapshot_dir"], data_cfg)
    rfm_dir = resolve_path(rfm_cfg["outputs"]["rfm_dir"], data_cfg)
    out_dir = resolve_path(cfg["outputs"]["features_dir"], data_cfg)
    tab_dir = resolve_path(cfg["outputs"]["tables_dir"], data_cfg)
    for d in (out_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    shapes, built = {}, {}
    for name, spec in snap_cfg["snapshots"].items():
        print(f"features {name}: building", flush=True)
        df = build_snapshot(con, base / name, rfm_dir / f"{name}.parquet", spec["cutoff"], cfg,
                            specs)
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        shapes[name], built[name] = (len(df), df.shape[1] - 1), df

    labels = pd.read_parquet(base / ref / "labels.parquet")
    uni = univariate(built[ref], labels, specs)
    flagged = uni[uni["strength"] > float(cfg["leakage_alarm_strength"])]
    if not flagged.empty:
        print(f"WARNING leakage alarm: {list(flagged['feature'])}", flush=True)
    uni.to_csv(tab_dir / f"univariate_{ref}.csv", index=False, float_format="%.10g")
    (tab_dir / "feature_registry.json").write_text(
        json.dumps([s.as_dict() for s in specs], indent=2), encoding="utf-8")

    results = resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg)
    write_section(results, "features", render_section(data_cfg, cfg, ref, specs, shapes, uni))
    dictionary = resolve_path(load_config("audit")["outputs"]["data_dictionary_md"], data_cfg)
    write_section(dictionary, "features", render_dictionary(specs, data_cfg),
                  header=DICTIONARY_HEADER)
    print(f"phase 5: {len(specs)} features -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
