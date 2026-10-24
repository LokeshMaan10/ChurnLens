"""Model inputs: feature selection from the registry, preprocessing, estimators.

Only registry features with use_in_model = true enter the model; msno and labels
never do. Preprocessing is fitted inside each CV fold (it lives in the pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnlens.features.registry import FeatureSpec

MISSING = "missing"
LOGREG_MAX_ITER = 2000


@dataclass
class Dataset:
    """Features, label and ids for one snapshot."""

    X: pd.DataFrame
    y: np.ndarray
    msno: pd.Series
    full: pd.DataFrame      # every feature column (incl. demographics) for the fairness audit


def model_columns(specs: list[FeatureSpec], include_demographics: bool = False
                  ) -> tuple[list[str], list[str]]:
    """(numeric/binary columns, categorical columns) allowed in the model."""
    use = [s for s in specs if s.use_in_model or (include_demographics
                                                  and s.group == "demographics")]
    return ([s.name for s in use if s.kind != "categorical"],
            [s.name for s in use if s.kind == "categorical"])


def load_dataset(features_file: Path, labels_file: Path, numeric: list[str],
                 categorical: list[str]) -> Dataset:
    """Join features to labels (one row per label user) and split X / y."""
    feats = pd.read_parquet(features_file)
    labels = pd.read_parquet(labels_file)
    df = labels.merge(feats, on="msno", how="left", validate="one_to_one")
    if len(df) != len(labels):
        raise AssertionError("feature/label row mismatch")
    return Dataset(prepare_X(df, numeric, categorical), df["is_churn"].to_numpy(dtype=int),
                   df["msno"], df)


def prepare_X(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Model input matrix: numeric as float (NaN kept), categorical as strings with NULL ->
    'missing'. Shared by training and by label-free live scoring."""
    X = df[numeric + categorical].copy()
    X[numeric] = X[numeric].astype(float)
    X[categorical] = X[categorical].astype(object).where(X[categorical].notna(), MISSING) \
        .astype(str)
    return X


def preprocessor(numeric: list[str], categorical: list[str], scale: bool, impute: bool,
                 min_frequency: float) -> ColumnTransformer:
    """Numeric: optional median impute (+ missing indicators) and scaling. Categorical:
    one-hot with rare and unseen categories grouped as infrequent."""
    num_steps: list[tuple[str, Any]] = []
    if impute:
        num_steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                  keep_empty_features=True)))
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num = Pipeline(num_steps) if num_steps else "passthrough"
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                        sparse_output=False)
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)],
                             verbose_feature_names_out=False)


def make_estimator(name: str, numeric: list[str], categorical: list[str], cfg: dict[str, Any]
                   ) -> Pipeline:
    """Pipeline(preprocess -> classifier) with class weights, for one model family."""
    seed, cw, mf = cfg["seed"], cfg["class_weight"], cfg["onehot_min_frequency"]
    fixed = dict(cfg["models"][name].get("fixed", {}))
    if name == "logreg":
        prep = preprocessor(numeric, categorical, scale=True, impute=True, min_frequency=mf)
        clf: Any = LogisticRegression(class_weight=cw, max_iter=LOGREG_MAX_ITER, random_state=seed)
    elif name == "random_forest":
        prep = preprocessor(numeric, categorical, scale=False, impute=True, min_frequency=mf)
        clf = RandomForestClassifier(class_weight=cw, random_state=seed, **fixed)
    elif name == "lightgbm":
        # LightGBM handles missing values natively: no imputation.
        prep = preprocessor(numeric, categorical, scale=False, impute=False, min_frequency=mf)
        clf = LGBMClassifier(class_weight=cw, random_state=seed, **fixed)
    else:
        raise ValueError(f"unknown model family {name}")
    return Pipeline([("prep", prep), ("clf", clf)])


def param_distributions(spec: dict[str, dict[str, list]]) -> dict[str, Any]:
    """Config search spaces -> sklearn distributions ({log_uniform: [a, b]} or {choice: [...]})."""
    out: dict[str, Any] = {}
    for name, d in spec.items():
        if "log_uniform" in d:
            out[name] = stats.loguniform(*d["log_uniform"])
        elif "choice" in d:
            out[name] = list(d["choice"])
        else:
            raise ValueError(f"unknown distribution for {name}: {d}")
    return out
