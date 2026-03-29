"""Predict yield using serialized model pipeline.

Usage:
    python scripts/predict.py --model data/processed/model_pipeline.joblib \
        --input data/processed/clean_data.csv --output data/processed/predictions.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from joblib import load as joblib_load

LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict yield using saved pipeline.")
    parser.add_argument(
        "--model",
        default=Path("data/processed/model_pipeline.joblib"),
        type=Path,
        help="Serialized model pipeline path",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input data CSV")
    parser.add_argument(
        "--output",
        default=Path("data/processed/predictions.csv"),
        type=Path,
        help="Output predictions CSV path",
    )
    parser.add_argument(
        "--metrics",
        default=Path("data/processed/model_metrics.json"),
        type=Path,
        help="Optional training metrics JSON containing features_used.",
    )
    parser.add_argument("--target", default="yield_tons", help="Target column name (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    model_pipeline = joblib_load(args.model)
    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("Input dataset is empty.")

    input_df = df.copy()
    if args.target in input_df.columns:
        input_df = input_df.drop(columns=[args.target])

    expected_features: list[str] = []
    if args.metrics.exists():
        try:
            metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
            if isinstance(metrics, dict) and isinstance(metrics.get("features_used"), list):
                expected_features = [str(x) for x in metrics["features_used"]]
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not parse metrics file at %s", args.metrics)

    if not expected_features and hasattr(model_pipeline, "feature_names_in_"):
        expected_features = [str(x) for x in list(model_pipeline.feature_names_in_)]

    if expected_features:
        missing = [c for c in expected_features if c not in input_df.columns]
        if missing:
            raise ValueError(
                "Input schema mismatch. Missing required feature columns: "
                + ", ".join(missing)
            )
        input_df = input_df[expected_features].copy()

    predictions = model_pipeline.predict(input_df)

    out_df = df.copy()
    out_df["predicted_yield_tons"] = predictions

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    LOGGER.info("Saved predictions to %s", args.output)


if __name__ == "__main__":
    main()
