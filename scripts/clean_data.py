"""Clean KoboToolbox survey data.

Rules applied:
- Drop Kobo metadata fields
- Normalize column names

Usage:
    python scripts/clean_data.py --input data/raw/kobo_export.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import ParserError

LOGGER = logging.getLogger(__name__)

KOBO_METADATA_FIELDS = {
    "start",
    "end",
    "today",
    "deviceid",
    "subscriberid",
    "simserial",
    "phonenumber",
    "username",
    "_id",
    "_uuid",
    "_submission_time",
    "_validation_status",
    "_notes",
    "_status",
    "_tags",
    "_version",
    "_submitted_by",
    "_xform_id_string",
    "_duration",
    "_index",
    "_attachments",
    "_geolocation",
}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_csv_robust(input_path: Path, encoding: str | None = None) -> pd.DataFrame:
    sample_encodings = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin1"]
    sample = ""
    detected_encoding = encoding
    last_decode_error: UnicodeDecodeError | None = None

    for enc in sample_encodings:
        try:
            with input_path.open("r", encoding=enc, newline="") as sample_file:
                sample = sample_file.read(4096)
            detected_encoding = enc
            if encoding is None and enc != "utf-8":
                LOGGER.info("Detected non-default encoding for sample read: %s", enc)
            break
        except UnicodeDecodeError as exc:
            last_decode_error = exc
            continue

    if not sample and last_decode_error is not None:
        raise last_decode_error

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","

    try:
        return pd.read_csv(input_path, encoding=detected_encoding, sep=sep)
    except ParserError:
        fallback_sep = ";" if sep != ";" else ","
        LOGGER.warning(
            "CSV parse failed with delimiter '%s'. Retrying with '%s'.",
            sep,
            fallback_sep,
        )
        return pd.read_csv(input_path, encoding=detected_encoding, sep=fallback_sep)


def normalize_column_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized or "field"


def normalize_columns(columns: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    normalized_columns: list[str] = []

    for column in columns:
        base = normalize_column_name(column)
        if base in seen:
            seen[base] += 1
            normalized = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            normalized = base
        normalized_columns.append(normalized)

    return normalized_columns


def drop_kobo_metadata(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols: set[str] = set()
    for column in df.columns:
        if column in KOBO_METADATA_FIELDS:
            drop_cols.add(column)
            continue
        if column.startswith("_"):
            drop_cols.add(column)
            continue
        if column.lower().startswith("meta/"):
            drop_cols.add(column)
            continue
    if drop_cols:
        LOGGER.info("Dropping Kobo metadata columns: %s", sorted(drop_cols))
    return df.drop(columns=list(drop_cols), errors="ignore")


def generate_report(original: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    return {
        "original_shape": list(original.shape),
        "cleaned_shape": list(cleaned.shape),
        "dropped_columns": sorted(set(original.columns) - set(cleaned.columns)),
        "columns": cleaned.columns.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean KoboToolbox survey data.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to raw Kobo CSV export.",
    )
    parser.add_argument(
        "--output",
        default=Path("data/processed/clean_data.csv"),
        type=Path,
        help="Output path for cleaned data.",
    )
    parser.add_argument(
        "--report",
        default=None,
        type=Path,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional file encoding if not UTF-8.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    raw_df = read_csv_robust(args.input, args.encoding)
    if raw_df.empty:
        raise ValueError("Input dataset is empty.")

    df = drop_kobo_metadata(raw_df)
    df.columns = normalize_columns(df.columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    LOGGER.info("Saved cleaned data to %s", args.output)

    if args.report:
        report = generate_report(raw_df, df)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        LOGGER.info("Wrote cleaning report to %s", args.report)


if __name__ == "__main__":
    main()
