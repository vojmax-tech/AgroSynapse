"""Train yield prediction model from clean survey data.

Usage:
    python scripts/train_model.py --input data/processed/clean_data.csv \
        --target yield_tons --output data/processed/model_pipeline.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
import numpy as np
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from joblib import dump as joblib_dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_config import APPROVED_FEATURES, select_allowlisted_features

LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def infer_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []

    for col in X.columns:
        series = X[col]
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return categorical_cols, numeric_cols


def select_training_features(df: pd.DataFrame, target: str, feature_allowlist: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    selected, risky_detected, blocked_from_dataset = select_allowlisted_features(
        df.columns.tolist(), target, feature_allowlist
    )
    if not selected:
        raise ValueError(
            "No approved feature columns were found in the input dataset. "
            "Update --features or ensure cleaned column names match the allowlist."
        )

    if risky_detected:
        LOGGER.warning("Dropping risky allowlisted columns: %s", risky_detected)

    if blocked_from_dataset:
        LOGGER.info("Blocked leakage-prone columns from dataset: %s", sorted(blocked_from_dataset))

    return df[selected].copy(), selected


def score_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n_folds": 0}
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std_v / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean_v, "std": std_v, "ci95": ci95, "n_folds": n}


def compute_cv_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    numeric_cols: list[str],
    n_jobs: int,
) -> dict[str, dict[str, float]] | None:
    n_samples = len(X)
    if n_samples < 6:
        return None

    n_splits = min(5, max(2, n_samples // 2))
    n_repeats = 2 if n_samples < 100 else 3
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    estimator = build_pipeline(categorical_cols, numeric_cols, n_jobs=n_jobs)
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=1,
        error_score="raise",
    )

    mae = -scores["test_mae"]
    rmse = np.sqrt(-scores["test_mse"])
    r2 = scores["test_r2"]
    return {
        "mae": score_summary(mae),
        "rmse": score_summary(rmse),
        "r2": score_summary(r2),
    }


def build_pipeline(
    categorical_cols: list[str], numeric_cols: list[str], n_jobs: int
) -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    transformers = []
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=n_jobs)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train yield prediction model.")
    parser.add_argument("--input", required=True, type=Path, help="Cleaned data CSV path")
    parser.add_argument("--target", default="yield_tons", help="Name of regression target")
    parser.add_argument(
        "--output",
        default=Path("data/processed/model_pipeline.joblib"),
        type=Path,
        help="Output path for serialized pipeline",
    )
    parser.add_argument(
        "--metrics",
        default=Path("data/processed/model_metrics.json"),
        type=Path,
        help="Output path for training metrics",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for RandomForest (use -1 for all cores).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size (0-1)")
    parser.add_argument(
        "--features",
        default=",".join(APPROVED_FEATURES),
        help=(
            "Comma-separated feature allowlist. "
            "Only these columns are used, after risky-column blocking."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("Input dataset is empty.")
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset.")

    initial_rows = len(df)
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df.dropna(subset=[args.target]).copy()
    dropped_target_rows = initial_rows - len(df)
    if dropped_target_rows > 0:
        LOGGER.info(
            "Dropped %s row(s) with non-numeric or missing target '%s'.",
            dropped_target_rows,
            args.target,
        )
    if len(df) < 4:
        raise ValueError(
            "Not enough valid rows after target filtering. "
            "Need at least 4 records with numeric target values."
        )

    feature_allowlist = [x.strip() for x in str(args.features).split(",") if x.strip()]
    X, selected_features = select_training_features(df, args.target, feature_allowlist)
    categorical_cols, numeric_cols = infer_feature_columns(X)

    if not categorical_cols and not numeric_cols:
        raise ValueError("No feature columns found after preprocessing.")

    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    pipeline = build_pipeline(categorical_cols, numeric_cols, n_jobs=args.n_jobs)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    cv_metrics = compute_cv_metrics(X, y, categorical_cols, numeric_cols, n_jobs=args.n_jobs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib_dump(pipeline, args.output)
    LOGGER.info("Saved model pipeline to %s", args.output)

    metrics = {
        "target": args.target,
        "test_size": args.test_size,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "features_used": selected_features,
        "n_features": len(selected_features),
        "cv": cv_metrics,
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Saved metrics to %s", args.metrics)

    LOGGER.info("Training complete: MAE=%.4f RMSE=%.4f R2=%.4f", mae, rmse, r2)


if __name__ == "__main__":
    main()
