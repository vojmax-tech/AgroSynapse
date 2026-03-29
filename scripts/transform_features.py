"""Transform cleaned data into model-ready features.

Rules applied:
- Encode categorical values (One-Hot)
- Impute missing values

Usage:
    python scripts/transform_features.py --input data/processed/clean_data.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_config import APPROVED_FEATURES, select_allowlisted_features

try:
    from joblib import dump as joblib_dump
except ImportError:  # pragma: no cover
    joblib_dump = None

LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def infer_feature_columns(df: pd.DataFrame, target: str | None) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[target], errors="ignore") if target else df
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []

    for column in feature_df.columns:
        series = feature_df[column]
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)

    return categorical_cols, numeric_cols


def build_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_encoder()),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    transformers = []
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform cleaned survey data.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to cleaned data CSV.",
    )
    parser.add_argument(
        "--target",
        default="yield_tons",
        help="Target column for modeling.",
    )
    parser.add_argument(
        "--features",
        default=",".join(APPROVED_FEATURES),
        help="Comma-separated feature allowlist to align with training.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/processed"),
        type=Path,
        help="Output directory for features and target.",
    )
    parser.add_argument(
        "--save-preprocessor",
        action="store_true",
        help="Persist fitted preprocessor as joblib.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("Input dataset is empty.")

    target_col = args.target if args.target in df.columns else None
    if args.target and target_col is None:
        LOGGER.warning(
            "Target column '%s' not found. Proceeding without target.",
            args.target,
        )

    feature_allowlist = [x.strip() for x in str(args.features).split(",") if x.strip()]
    target_name_for_selection = target_col if target_col is not None else args.target
    selected, risky_in_allowlist, blocked_from_dataset = select_allowlisted_features(
        df.columns.tolist(), target_name_for_selection, feature_allowlist
    )
    if not selected:
        raise ValueError(
            "No approved feature columns were found in the input dataset. "
            "Update --features or check cleaned column names."
        )
    if risky_in_allowlist:
        LOGGER.warning("Dropping risky allowlisted columns: %s", risky_in_allowlist)
    if blocked_from_dataset:
        LOGGER.info("Blocked leakage-prone columns from dataset: %s", sorted(blocked_from_dataset))

    feature_df = df[selected].copy()
    categorical_cols, numeric_cols = infer_feature_columns(feature_df, target=None)
    if not categorical_cols and not numeric_cols:
        raise ValueError("No feature columns found after preprocessing.")

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    features = preprocessor.fit_transform(feature_df)

    feature_names = preprocessor.get_feature_names_out()
    features_df = pd.DataFrame(features, columns=feature_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.output_dir / "features.csv"
    features_df.to_csv(features_path, index=False)

    LOGGER.info("Saved features to %s", features_path)

    if target_col:
        target_path = args.output_dir / "target.csv"
        df[[target_col]].to_csv(target_path, index=False)
        LOGGER.info("Saved target to %s", target_path)

    if args.save_preprocessor:
        if joblib_dump is None:
            raise RuntimeError("joblib is required to save the preprocessor.")
        preprocessor_path = args.output_dir / "preprocessor.joblib"
        joblib_dump(preprocessor, preprocessor_path)
        LOGGER.info("Saved preprocessor to %s", preprocessor_path)


if __name__ == "__main__":
    main()
