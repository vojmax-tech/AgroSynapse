"""Orchestrate ETL + train + predict steps.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --input data/raw/AgroSynapse_Survey_.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add scripts directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from feature_config import APPROVED_FEATURES, is_risky_column, select_allowlisted_features


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def parse_feature_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def infer_target_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None

    preferred_tokens = [
        "target",
        "yield",
        "price",
        "usdprice",
        "value",
        "amount",
        "cost",
        "sales",
        "revenue",
        "score",
    ]
    depriority_tokens = ["lat", "lon", "latitude", "longitude", "year", "month", "day", "id", "index"]

    numeric_cols: list[str] = []
    for col in df.columns:
        if is_risky_column(col):
            continue
        ratio = float(pd.to_numeric(df[col], errors="coerce").notna().mean())
        if ratio >= 0.5:
            numeric_cols.append(col)

    if not numeric_cols:
        return None

    def score(col: str) -> float:
        norm = col.strip().lower()
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        ratio = float(numeric_series.notna().mean())
        variability = float(numeric_series.nunique(dropna=True))
        token_bonus = 30.0 if any(tok in norm for tok in preferred_tokens) else 0.0
        token_penalty = -22.0 if any(tok in norm for tok in depriority_tokens) else 0.0
        return (ratio * 100.0) + min(variability, 1000.0) / 20.0 + token_bonus + token_penalty

    return max(numeric_cols, key=score)


def pick_target_column(columns: list[str], preferred_target: str, df: pd.DataFrame | None = None) -> str | None:
    preferred = preferred_target.strip()
    if preferred and preferred.lower() != "auto" and preferred in columns:
        return preferred

    fallback_order = [
        "yield_tons",
        "yield",
        "yield_kg",
        "actual_yield_tons",
        "price",
        "usdprice",
        "value",
    ]
    for candidate in fallback_order:
        if candidate in columns:
            return candidate

    if df is not None:
        inferred = infer_target_column(df)
        if inferred in columns:
            return inferred

    return None


def auto_feature_candidates(columns: list[str], target: str | None) -> list[str]:
    return [col for col in columns if col != target and not is_risky_column(col)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help=(
            "Fail when approved allowlisted features or target are missing. "
            "Default mode is tolerant and auto-selects safe fallback features."
        ),
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
            sys.executable,
            "scripts/ingest_kobo.py",
            "--input",
            str(data_raw),
            "--output-dir",
            str(data_ingested.parent),
            "--output-file",
            data_ingested.name,
        ]
    )
    run([sys.executable, "scripts/clean_data.py", "--input", str(data_ingested), "--output", str(data_clean)])

    if not data_clean.exists():
        raise FileNotFoundError(f"Cleaned data not found after cleaning: {data_clean}")
    clean_df = pd.read_csv(data_clean)
    if clean_df.empty:
        raise ValueError("Cleaned dataset is empty after cleaning.")

    available_columns = clean_df.columns.tolist()
    resolved_target = pick_target_column(available_columns, args.target, clean_df)
    if resolved_target is None:
        if args.strict_schema:
            raise ValueError(
                f"Target column '{args.target}' was not found and no fallback target exists in cleaned data."
            )
        print(
            f"Warning: target column '{args.target}' not found. "
            "Training and prediction will be skipped for this run."
        )
    elif args.target.strip().lower() == "auto":
        print(f"Info: auto-detected target column '{resolved_target}'.")
    elif resolved_target != args.target:
        print(f"Info: target '{args.target}' not found. Using fallback target '{resolved_target}'.")

    requested_features = parse_feature_list(args.features)
    target_for_selection = resolved_target if resolved_target is not None else args.target
    selected_features, risky_in_allowlist, blocked_from_dataset = select_allowlisted_features(
        available_columns, target_for_selection, requested_features
    )
    if risky_in_allowlist:
        print(f"Info: dropping risky allowlisted features: {', '.join(risky_in_allowlist)}")
    if blocked_from_dataset:
        print(f"Info: blocked risky dataset columns: {', '.join(sorted(blocked_from_dataset))}")

    if not selected_features:
        if args.strict_schema:
            raise ValueError(
                "No approved feature columns were found in the cleaned dataset while --strict-schema is enabled."
            )
        selected_features = auto_feature_candidates(available_columns, resolved_target)
        if selected_features:
            print(
                "Info: no approved allowlisted features matched. "
                f"Using {len(selected_features)} safe auto-selected feature(s)."
            )

    if not selected_features:
        raise ValueError(
            "No usable feature columns were found after filtering. "
            "Dataset may only contain blocked/risky columns."
        )

    selected_features_csv = ",".join(selected_features)
    if resolved_target is None:
        features_output_dir.mkdir(parents=True, exist_ok=True)
        fallback_features_path = features_output_dir / "features.csv"
        clean_df[selected_features].copy().to_csv(fallback_features_path, index=False)
        print(
            "Info: target missing, so one-hot transform was skipped to avoid heavy memory use. "
            f"Wrote lightweight features to {fallback_features_path}."
        )
        write_json(
            metrics_output,
            {
                "status": "skipped",
                "reason": f"Target column '{args.target}' not found in cleaned dataset.",
                "n_rows": int(clean_df.shape[0]),
                "n_columns": int(clean_df.shape[1]),
                "features_used": selected_features,
                "n_features": len(selected_features),
                "feature_transform": "skipped_no_target_lightweight_features",
            },
        )
        print("Pipeline completed partially: ingest + clean succeeded; model/predict skipped.")
        return

    run(
        [
            sys.executable,
            "scripts/transform_features.py",
            "--input",
            str(data_clean),
            "--target",
            resolved_target,
            "--output-dir",
            str(features_output_dir),
            "--features",
            selected_features_csv,
        ]
    )

    run(
        [
            sys.executable,
            "scripts/train_model.py",
            "--input",
            str(data_clean),
            "--target",
            resolved_target,
            "--output",
            str(model_output),
            "--metrics",
            str(metrics_output),
            "--n-jobs",
            str(args.n_jobs),
            "--features",
            selected_features_csv,
        ]
    )
    run(
        [
            sys.executable,
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
            resolved_target,
        ]
    )

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
