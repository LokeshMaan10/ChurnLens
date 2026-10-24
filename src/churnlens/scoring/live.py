"""Live scoring of April 2017 users (snapshot C) — the only consumer of scoring-only logs.

Run with `python -m churnlens.scoring.live` (or `make live-score`). Builds a label-free
snapshot directory with the same layout as A/B (inputs strictly before C's cutoff), reuses
the Phase 4 scorer and Phase 5 feature builder, and scores with the Phase 6 model. Value is
the formula CLTV (the Phase 8 Cox model object is not persisted). Output:
data/processed/{mode}/scoring/C_scores.parquet. Never used for training or evaluation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from churnlens.config import load_config, resolve_path
from churnlens.features.build import build_snapshot
from churnlens.features.registry import all_specs
from churnlens.modeling.data import prepare_X
from churnlens.reporting import fmt_int, md_table, run_banner, write_section
from churnlens.segmentation import rfm
from churnlens.value import cltv


def build_inputs(con: duckdb.DuckDBPyConnection, out: Path, cfg: dict) -> int:
    """Labels-free snapshot C: population, transactions, logs (training + scoring-only) and
    members, every input strictly before the cutoff."""
    c = f"DATE '{cfg['cutoff']}'"
    if out.exists():
        shutil.rmtree(out)
    (out / "user_logs").mkdir(parents=True)
    con.execute(f"""CREATE OR REPLACE TEMP VIEW pop AS
        SELECT msno FROM transactions
        WHERE is_cancel = 0 AND transaction_date < {c}
        GROUP BY msno
        HAVING MAX(membership_expire_date) BETWEEN DATE '{cfg['expiry_month_start']}'
                                               AND DATE '{cfg['expiry_month_end']}'""")
    users = "msno IN (SELECT msno FROM pop)"
    copies = {
        "labels.parquet": "SELECT msno, CAST(NULL AS TINYINT) AS is_churn FROM pop",
        "transactions.parquet": f"SELECT * FROM transactions WHERE {users} "
                                f"AND transaction_date < {c}",
        "members.parquet": f"SELECT * EXCLUDE (registration_date), CASE WHEN registration_date "
                           f"< {c} THEN registration_date END AS registration_date, "
                           f"registration_date >= {c} AS registered_after_cutoff FROM members "
                           f"WHERE {users}",
    }
    for name, sql in copies.items():
        con.execute(f"COPY ({sql}) TO '{(out / name).as_posix()}' (FORMAT parquet)")
    # Scoring-only March logs are combined with the training logs HERE ONLY.
    con.execute(f"""COPY (
        SELECT *, strftime(date, '%Y-%m') AS month FROM (
            SELECT * FROM user_logs UNION ALL SELECT * FROM user_logs_scoring)
        WHERE {users} AND date < {c}
        ) TO '{(out / 'user_logs').as_posix()}' (FORMAT parquet, PARTITION_BY (month))""")
    return con.execute("SELECT COUNT(*) FROM pop").fetchone()[0]


def score_rfm(snap: Path, cutoff, out_file: Path) -> None:
    """Phase 4 scores, segments and groups using snapshot A's saved references and model."""
    cfg = load_config("rfm")
    data_cfg = load_config("data")
    tab = resolve_path(cfg["outputs"]["tables_dir"], data_cfg)
    refs = {k: rfm.ScoreReference.from_json(v) for k, v in json.loads(
        (tab / "score_reference.json").read_text(encoding="utf-8")).items()}
    raw = rfm.raw_rfm(duckdb.connect(), snap, cutoff, cfg)
    df = rfm.score(raw, refs, int(cfg["n_levels"]))
    df["segment"] = rfm.assign_segments(df, cfg["segments"])
    df["personal_recency_band"] = df["personal_recency"].map(rfm.personal_recency_band)
    km = json.loads((tab / "kmeans_model.json").read_text(encoding="utf-8"))
    from churnlens.segmentation.kmeans import excluded_labels, feature_matrix, in_population
    pop = in_population(raw, km["population"])
    labels = excluded_labels(raw, cfg["kmeans"]["excluded_groups"]).astype(object)
    if pop.any():
        x = (feature_matrix(raw[pop]) - np.array(km["scaler_mean"])) / np.array(km["scaler_scale"])
        cent = np.array(km["centroids"])
        nearest = np.argmin(((x[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2), axis=1)
        labels[pop] = [km["label_map"][str(int(i))] for i in nearest]
    df["cluster"] = labels
    df.to_parquet(out_file, index=False)


def main() -> int:
    """Build C, featurize, score, value, report."""
    data_cfg, cfg = load_config("data"), load_config("scoring")
    model_cfg, value_cfg, clean_cfg = (load_config("model"), load_config("value"),
                                       load_config("cleaning"))
    base = resolve_path(cfg["outputs"]["scoring_dir"], data_cfg)
    snap = base / cfg["name"]
    con = duckdb.connect(str(resolve_path(clean_cfg["outputs"]["database"], data_cfg)),
                         read_only=True)
    n = build_inputs(con, snap, cfg)
    con.close()
    cutoff = pd.Timestamp(cfg["cutoff"]).date()
    rfm_file = base / f"{cfg['name']}_rfm.parquet"
    score_rfm(snap, cutoff, rfm_file)
    feat_cfg = load_config("features")
    feats = build_snapshot(duckdb.connect(), snap, rfm_file, cutoff, feat_cfg,
                           all_specs(feat_cfg))
    bundle = joblib.load(resolve_path(model_cfg["outputs"]["model_dir"], data_cfg)
                         / "churn_model.joblib")
    X = prepare_X(feats, bundle["numeric"], bundle["categorical"])
    p_raw = bundle["pipeline"].predict_proba(X)[:, 1]
    p = bundle["calibrator"].predict(p_raw)
    rates = pd.read_csv(resolve_path(value_cfg["outputs"]["tables_dir"], data_cfg)
                        / "formula_segment_rates.csv").set_index("segment")["renewal_rate"]
    seg = cltv.segment_key(feats, value_cfg["formula_segments"]["plan_buckets"])
    price = cltv.monthly_price(feats, float(feats.loc[feats["last_amount_paid"] > 0,
                                                     "last_amount_paid"].median()),
                               int(value_cfg["days_per_month"]))
    renewal = seg.map(rates).fillna(float(rates.mean())).to_numpy()
    c = cltv.cltv_from_survival((price * float(value_cfg["margin"])).to_numpy(),
                                cltv.formula_survival(renewal, int(value_cfg["horizon_months"])),
                                cltv.monthly_discount(float(value_cfg["annual_discount_rate"])))
    t = float(bundle["threshold"])
    scores = pd.DataFrame({"msno": feats["msno"], "p_raw": p_raw, "p_churn": p,
                           "risk_band": np.where(p >= t, "high", "low"), "cltv_formula": c,
                           "evar": p * c, "rfm_segment": feats["rfm_segment"],
                           "last_is_auto_renew": feats["last_is_auto_renew"]}).round(8)
    scores.to_parquet(base / f"{cfg['name']}_scores.parquet", index=False)
    parts = ["## Phase 15: live scoring demo (snapshot C, April 2017 expirations)", "",
             run_banner(data_cfg), "",
             f"Scoring-only snapshot **{cfg['name']}** (cutoff {cfg['cutoff']}): "
             f"{fmt_int(n)} sampled users whose latest pre-cutoff membership expires in April "
             "2017. Features use transactions and logs strictly before the cutoff, including "
             "the scoring-only March 2017 logs (their only use). No labels exist, so there is "
             "no accuracy here; value uses the formula CLTV.", ""]
    parts.append(md_table(["Measure", "Value"], [
        ["Users scored", fmt_int(len(scores))],
        ["Mean calibrated P(churn)", f"{scores['p_churn'].mean():.2%}"],
        [f"High risk (P ≥ {t:.3f})", f"{fmt_int(int((scores['risk_band'] == 'high').sum()))} "
                                     f"({(scores['risk_band'] == 'high').mean():.1%})"],
        ["Total expected value at risk (NTD, placeholder margin)",
         fmt_int(int(round(scores['evar'].sum())))],
    ], "lr"))
    write_section(resolve_path(load_config("audit")["outputs"]["results_md"], data_cfg),
                  "live_scoring", "\n".join(parts))
    print(f"live scoring: {len(scores):,} users; mean P {scores['p_churn'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
