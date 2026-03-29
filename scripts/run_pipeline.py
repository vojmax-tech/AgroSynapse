"""Orchestrate ETL + train + predict steps.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --input data/raw/AgroSynapse_Survey_.csv
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from feature_config import APPROVED_FEATURES


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full AgroSynapse pipeline.")
    parser.add_argument(
        "--input",
        default=Path("data/raw/kobo_export.csv"),
        type=Path,
        help="Path to raw Kobo export CSV.",
    )
    parser.add_argument(
        "--ingested-output",
        default=Path("data/raw/kobo_export_raw.csv"),
        type=Path,
        help="Path for ingested raw CSV used by downstream steps.",
    )
    parser.add_argument(
        "--clean-output",
        default=Path("data/processed/clean_data.csv"),
        type=Path,
        help="Path for cleaned dataset output.",
    )
    parser.add_argument(
        "--features-output-dir",
        default=Path("data/processed"),
        type=Path,
        help="Directory where engineered features.csv will be written.",
    )
    parser.add_argument(
        "--model-output",
        default=Path("data/processed/model_pipeline.joblib"),
        type=Path,
        help="Path for serialized model pipeline.",
    )
    parser.add_argument(
        "--metrics-output",
        default=Path("data/processed/model_metrics.json"),
        type=Path,
        help="Path for model metrics JSON.",
    )
    parser.add_argument(
        "--predictions-output",
        default=Path("data/processed/predictions.csv"),
        type=Path,
        help="Path for predictions CSV.",
    )
    parser.add_argument(
        "--target",
        default="yield_tons",
        help="Target column used for training and optional prediction drop.",
    )
    parser.add_argument(
        "--features",
        default=",".join(APPROVED_FEATURES),
        help="Comma-separated approved feature list shared by transform/train.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_raw = args.input
    data_ingested = args.ingested_output
    data_clean = args.clean_output
    features_output_dir = args.features_output_dir
    model_output = args.model_output
    metrics_output = args.metrics_output
    predictions_output = args.predictions_output

    run(
        [
            "python",
            "scripts/ingest_kobo.py",
            "--input",
            str(data_raw),
            "--output-dir",
            str(data_ingested.parent),
            "--output-file",
            data_ingested.name,
        ]
    )
    run(["python", "scripts/clean_data.py", "--input", str(data_ingested), "--output", str(data_clean)])
    run(
        [
            "python",
            "scripts/transform_features.py",
            "--input",
            str(data_clean),
            "--target",
            args.target,
            "--output-dir",
            str(features_output_dir),
            "--features",
            args.features,
        ]
    )
    run(
        [
            "python",
            "scripts/train_model.py",
            "--input",
            str(data_clean),
            "--target",
            args.target,
            "--output",
            str(model_output),
            "--metrics",
            str(metrics_output),
            "--n-jobs",
            str(args.n_jobs),
            "--features",
            args.features,
        ]
    )
    run(
        [
            "python",
            "scripts/predict.py",
            "--model",
            str(model_output),
            "--input",
            str(data_clean),
            "--output",
            str(predictions_output),
            "--metrics",
            str(metrics_output),
            "--target",
            args.target,
        ]
    )

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
