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
from transform_utils import ToNumericTransformer, ToStringTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_config import APPROVED_FEATURES, resolve_feature_columns

LOGGER = logging.getLogger(__name__)

MIN_VALID_TARGET_ROWS = 4
LARGE_DATASET_ROWS = 100_000
HIGH_CARDINALITY_LIMIT = 50
HIGH_ESTIMATED_ENCODED_FEATURES = 2_000
CV_REPEAT_ROW_LIMIT = 5_000
MIN_TEMPORAL_PERIODS = 6


def make_one_hot_encoder(
    sparse_output: bool = True,
    min_frequency: int | None = None,
    max_categories: int | None = None,
) -> OneHotEncoder:
    kwargs: dict[str, object] = {
        "handle_unknown": "ignore",
        "sparse_output": sparse_output,
    }
    if min_frequency is not None:
        kwargs["min_frequency"] = min_frequency
    if max_categories is not None:
        kwargs["max_categories"] = max_categories
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        kwargs["sparse"] = kwargs.pop("sparse_output")
        return OneHotEncoder(**kwargs)




def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def infer_feature_columns(X: pd.DataFrame, numeric_ratio_threshold: float = 0.85) -> tuple[list[str], list[str]]:
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []

    for col in X.columns:
        series = X[col]
        numeric_ratio = float(pd.to_numeric(series, errors="coerce").notna().mean())
        if (is_numeric_dtype(series) or numeric_ratio >= numeric_ratio_threshold) and not is_bool_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return categorical_cols, numeric_cols


def select_training_features(
    df: pd.DataFrame, target: str, feature_allowlist: list[str]
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    decision = resolve_feature_columns(df.columns.tolist(), target, feature_allowlist)
    selected = list(decision["selected"])
    if not selected:
        raise ValueError(
            "No usable feature columns were found in the input dataset. Every column was "
            "either the target, an identifier, or derived from the target."
        )

    if decision["fallback_used"]:
        LOGGER.warning("%s", decision["reason"])
    else:
        LOGGER.info("%s", decision["reason"])

    if decision["risky_in_allowlist"]:
        LOGGER.warning("Dropping risky allowlisted columns: %s", decision["risky_in_allowlist"])

    if decision["blocked"]:
        for column, why in sorted(decision["blocked"].items()):
            LOGGER.info("Blocked '%s' from features (%s).", column, why)

    LOGGER.info("Training on %s feature column(s): %s", len(selected), ", ".join(selected))
    return df[selected].copy(), selected, decision


def score_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n_folds": 0}
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std_v / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean_v, "std": std_v, "ci95": ci95, "n_folds": n}


def dataset_size_profile(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> dict[str, object]:
    categorical_cardinality = {
        col: int(X[col].astype("string").fillna("__missing__").nunique())
        for col in categorical_cols
    }
    estimated_encoded_features = int(sum(categorical_cardinality.values()) + len(numeric_cols))
    high_cardinality_cols = [
        col for col, count in categorical_cardinality.items() if count >= HIGH_CARDINALITY_LIMIT
    ]

    warnings_list: list[str] = []
    if len(X) < 30:
        warnings_list.append("Very small training set; metrics may vary heavily between runs.")
    if len(X) >= LARGE_DATASET_ROWS:
        warnings_list.append("Large dataset detected; training uses capped category encoding.")
    if high_cardinality_cols:
        warnings_list.append(
            "High-cardinality categorical fields detected: "
            + ", ".join(high_cardinality_cols[:6])
            + ("..." if len(high_cardinality_cols) > 6 else "")
        )
    if estimated_encoded_features >= HIGH_ESTIMATED_ENCODED_FEATURES:
        warnings_list.append("Estimated encoded feature count is high; rare categories will be grouped.")
    if y.nunique(dropna=True) <= 2:
        warnings_list.append("Target has very low variation; regression metrics may be less informative.")

    return {
        "n_rows": int(len(X)),
        "n_columns": int(X.shape[1]),
        "n_numeric_features": int(len(numeric_cols)),
        "n_categorical_features": int(len(categorical_cols)),
        "estimated_encoded_features": estimated_encoded_features,
        "high_cardinality_features": high_cardinality_cols,
        "warnings": warnings_list,
    }


def encoder_limits(profile: dict[str, object]) -> tuple[int | None, int | None]:
    n_rows = int(profile["n_rows"])
    estimated_encoded = int(profile["estimated_encoded_features"])
    if n_rows >= LARGE_DATASET_ROWS or estimated_encoded >= HIGH_ESTIMATED_ENCODED_FEATURES:
        return (5, 60)
    if estimated_encoded >= 800:
        return (3, 80)
    return (None, None)


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
    # Repeats buy stability on small samples; on tens of thousands of rows they only buy
    # training time, so drop to a single pass once the folds are already large.
    if n_samples < 100:
        n_repeats = 2
    elif n_samples < CV_REPEAT_ROW_LIMIT:
        n_repeats = 3
    else:
        n_repeats = 1
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    profile = dataset_size_profile(X, y, categorical_cols, numeric_cols)
    min_frequency, max_categories = encoder_limits(profile)
    estimator = build_pipeline(
        categorical_cols,
        numeric_cols,
        n_jobs=n_jobs,
        model_kind="random_forest",
        min_frequency=min_frequency,
        max_categories=max_categories,
        n_rows=n_samples,
    )
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


def forest_capacity(n_rows: int) -> tuple[int, int]:
    """Pick (n_estimators, min_samples_leaf) so the saved model stays a sane size.

    Fully grown forests store roughly two nodes per training row per tree. On the ~20k-row
    demo panel, 240 unconstrained trees serialise to several hundred megabytes - too large
    to commit and slow to load on a small host - while adding nothing over leafier trees,
    because single-row leaves on a country/crop panel are memorisation rather than signal.
    """
    if n_rows >= 20_000:
        return 80, 20
    if n_rows >= 5_000:
        return 120, 10
    if n_rows >= 1_000:
        return 200, 4
    return 240, 1


def build_pipeline(
    categorical_cols: list[str],
    numeric_cols: list[str],
    n_jobs: int,
    model_kind: str = "random_forest",
    min_frequency: int | None = None,
    max_categories: int | None = None,
    n_rows: int = 0,
) -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_string", ToStringTransformer()),
            (
                "encoder",
                make_one_hot_encoder(
                    sparse_output=True,
                    min_frequency=min_frequency,
                    max_categories=max_categories,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("to_numeric", ToNumericTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    transformers = []
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    n_estimators, min_samples_leaf = forest_capacity(n_rows)
    if model_kind == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=n_jobs,
            min_samples_leaf=min_samples_leaf,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=n_jobs,
            min_samples_leaf=min_samples_leaf,
        )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def detect_time_column(X: pd.DataFrame, explicit: str | None = None) -> str | None:
    """Find a column that orders rows in time, so the model can be scored forward."""
    if explicit:
        return explicit if explicit in X.columns else None
    for candidate in ("year", "season_year", "harvest_year", "period"):
        if candidate in X.columns:
            values = pd.to_numeric(X[candidate], errors="coerce").dropna()
            if values.nunique() >= MIN_TEMPORAL_PERIODS:
                return candidate
    return None


def temporal_holdout_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    time_column: str,
    categorical_cols: list[str],
    numeric_cols: list[str],
    n_jobs: int,
    profile: dict[str, object],
    model_kind: str,
    holdout_fraction: float = 0.2,
) -> dict[str, object] | None:
    """Train on earlier periods and score on the most recent ones.

    A random split of a country/crop/year panel puts 2019 and 2020 rows for the same
    country and crop on opposite sides of the split, which is close to memorising a
    neighbour rather than predicting. Holding out the final periods measures the thing a
    user actually wants: how the model does on a season it has never seen.
    """
    periods = pd.to_numeric(X[time_column], errors="coerce")
    unique_periods = np.sort(periods.dropna().unique())
    if unique_periods.size < MIN_TEMPORAL_PERIODS:
        return None

    n_holdout = max(1, int(round(unique_periods.size * holdout_fraction)))
    cutoff = unique_periods[-n_holdout]
    train_mask = periods < cutoff
    test_mask = periods >= cutoff
    if train_mask.sum() < MIN_VALID_TARGET_ROWS or test_mask.sum() < 1:
        return None

    min_frequency, max_categories = encoder_limits(profile)
    pipeline = build_pipeline(
        categorical_cols,
        numeric_cols,
        n_jobs=n_jobs,
        model_kind=model_kind,
        min_frequency=min_frequency,
        max_categories=max_categories,
        n_rows=int(train_mask.sum()),
    )
    pipeline.fit(X[train_mask], y[train_mask])
    model_metrics = regression_metrics(y[test_mask], pipeline.predict(X[test_mask]))

    baseline = DummyRegressor(strategy="median").fit(X[train_mask], y[train_mask])
    baseline_metrics = regression_metrics(y[test_mask], baseline.predict(X[test_mask]))

    return {
        "time_column": time_column,
        "train_periods": [float(unique_periods[0]), float(cutoff - 1)],
        "test_periods": [float(cutoff), float(unique_periods[-1])],
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "model_type": model_kind,
        **model_metrics,
        "baseline_metrics": baseline_metrics,
        "persistence_metrics": persistence_baseline_metrics(
            X, y, train_mask, test_mask, categorical_cols
        ),
    }


def persistence_baseline_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: pd.Series,
    test_mask: pd.Series,
    categorical_cols: list[str],
) -> dict[str, float] | None:
    """Score "carry the last observed value forward" on the same forward-looking split.

    A median baseline is far too easy to beat on a panel: every series has its own level,
    so any model that learns those levels looks excellent. The baseline worth reporting is
    the one a practitioner would actually use - last season's value for this same series.
    A model that cannot beat it is not adding forecasting value, however high its R2.
    """
    if not categorical_cols:
        return None

    grouped = pd.concat([X[categorical_cols].astype(str), y.rename("__target__")], axis=1)
    last_seen = grouped[train_mask].groupby(categorical_cols, observed=True)["__target__"].last()

    test_keys = pd.MultiIndex.from_frame(X.loc[test_mask, categorical_cols].astype(str))
    if last_seen.index.nlevels != test_keys.nlevels:
        return None

    predictions = pd.Series(last_seen.reindex(test_keys).to_numpy(), index=y[test_mask].index)
    coverage = float(predictions.notna().mean())
    # Series never seen during training fall back to the training median, as a real
    # carry-forward forecast would have to.
    predictions = predictions.fillna(float(y[train_mask].median()))

    metrics = regression_metrics(y[test_mask], predictions.to_numpy())
    metrics["coverage"] = round(coverage, 4)
    metrics["group_columns"] = list(categorical_cols)
    return metrics


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def candidate_model_kinds(profile: dict[str, object]) -> list[str]:
    if int(profile["n_rows"]) < 12:
        return ["random_forest"]
    return ["random_forest", "extra_trees"]


def select_best_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_cols: list[str],
    numeric_cols: list[str],
    n_jobs: int,
    profile: dict[str, object],
) -> tuple[Pipeline, str, dict[str, dict[str, float]], dict[str, float]]:
    min_frequency, max_categories = encoder_limits(profile)
    candidate_metrics: dict[str, dict[str, float]] = {}
    best_pipeline: Pipeline | None = None
    best_kind = "random_forest"
    best_mae = float("inf")

    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)
    baseline_metrics = regression_metrics(y_test, baseline.predict(X_test))

    for model_kind in candidate_model_kinds(profile):
        pipeline = build_pipeline(
            categorical_cols,
            numeric_cols,
            n_jobs=n_jobs,
            model_kind=model_kind,
            min_frequency=min_frequency,
            max_categories=max_categories,
            n_rows=len(X_train),
        )
        pipeline.fit(X_train, y_train)
        metrics = regression_metrics(y_test, pipeline.predict(X_test))
        candidate_metrics[model_kind] = metrics
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_kind = model_kind
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model candidate could be trained.")

    return best_pipeline, best_kind, candidate_metrics, baseline_metrics


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
        "--time-column",
        default=None,
        help=(
            "Column ordering rows in time. When given (or auto-detected), metrics also "
            "include a forward-looking holdout on the most recent periods."
        ),
    )
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
    if len(df) < MIN_VALID_TARGET_ROWS:
        raise ValueError(
            "Not enough valid rows after target filtering. "
            "Need at least 4 records with numeric target values."
        )

    feature_allowlist = [x.strip() for x in str(args.features).split(",") if x.strip()]
    X, selected_features, feature_decision = select_training_features(
        df, args.target, feature_allowlist
    )
    categorical_cols, numeric_cols = infer_feature_columns(X)

    if not categorical_cols and not numeric_cols:
        raise ValueError("No feature columns found after preprocessing.")

    y = df[args.target]
    profile = dataset_size_profile(X, y, categorical_cols, numeric_cols)
    for warning in profile["warnings"]:
        LOGGER.warning("%s", warning)
    min_frequency, max_categories = encoder_limits(profile)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    pipeline, selected_model, candidate_metrics, baseline_metrics = select_best_pipeline(
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_cols,
        numeric_cols,
        args.n_jobs,
        profile,
    )
    selected_metrics = candidate_metrics[selected_model]
    mae = selected_metrics["mae"]
    rmse = selected_metrics["rmse"]
    r2 = selected_metrics["r2"]
    cv_metrics = compute_cv_metrics(X, y, categorical_cols, numeric_cols, n_jobs=args.n_jobs)

    time_column = detect_time_column(X, args.time_column)
    temporal_metrics = None
    if time_column:
        temporal_metrics = temporal_holdout_metrics(
            X, y, time_column, categorical_cols, numeric_cols, args.n_jobs, profile, selected_model
        )
        if temporal_metrics:
            LOGGER.info(
                "Forward-looking holdout on '%s' (test %s-%s): MAE=%.2f R2=%.3f",
                time_column,
                int(temporal_metrics["test_periods"][0]),
                int(temporal_metrics["test_periods"][1]),
                temporal_metrics["mae"],
                temporal_metrics["r2"],
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Forests of decision trees are highly compressible (mostly repeated int32 arrays):
    # level 3 cuts the artefact to about a third with no measurable effect on load time,
    # which matters when the file is committed and deployed.
    joblib_dump(pipeline, args.output, compress=3)
    LOGGER.info(
        "Saved model pipeline to %s (%.1f MB)",
        args.output,
        args.output.stat().st_size / 1_048_576,
    )

    metrics = {
        "target": args.target,
        "test_size": args.test_size,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "model_type": selected_model,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "dataset_profile": profile,
        "forest": dict(
            zip(("n_estimators", "min_samples_leaf"), forest_capacity(int(X_train.shape[0])))
        ),
        "encoder": {
            "sparse_output": True,
            "min_frequency": min_frequency,
            "max_categories": max_categories,
        },
        "features_used": selected_features,
        "n_features": len(selected_features),
        "feature_selection": feature_decision,
        "cv": cv_metrics,
        "temporal_holdout": temporal_metrics,
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Saved metrics to %s", args.metrics)

    LOGGER.info("Training complete: MAE=%.4f RMSE=%.4f R2=%.4f", mae, rmse, r2)


if __name__ == "__main__":
    main()
