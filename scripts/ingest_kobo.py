"""Ingest KoboToolbox exports into the raw data zone.

Usage:
    python scripts/ingest_kobo.py --input data/raw/kobo_export.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError

LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ingest_csv(input_path: Path, encoding: str | None = None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    LOGGER.info("Reading Kobo export from %s", input_path)
    # Kobo exports are often semicolon-delimited in some locales.
    # Detect delimiter from file sample and use a robust fallback sequence.
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
        df = pd.read_csv(input_path, encoding=detected_encoding, sep=sep)
    except ParserError:
        fallback_sep = ";" if sep != ";" else ","
        LOGGER.warning(
            "CSV parse failed with delimiter '%s'. Retrying with '%s'.",
            sep,
            fallback_sep,
        )
        df = pd.read_csv(input_path, encoding=detected_encoding, sep=fallback_sep)

    if df.empty:
        raise ValueError("Input dataset is empty.")

    return df


def resolve_output_path(
    output_dir: Path,
    output_file: str | None,
    input_path: Path,
    timestamp_output: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_file:
        return output_dir / output_file

    stem = input_path.stem.replace(" ", "_").lower()
    if timestamp_output:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return output_dir / f"{stem}_raw_{timestamp}.csv"
    return output_dir / f"{stem}_raw.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest KoboToolbox export data.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to Kobo CSV export.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/raw"),
        type=Path,
        help="Destination directory for raw data.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output file name. Defaults to <input_stem>_raw.csv.",
    )
    parser.add_argument(
        "--timestamp-output",
        action="store_true",
        help="When set, append a UTC timestamp to the default output filename.",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional file encoding if not UTF-8.",
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

    df = ingest_csv(args.input, args.encoding)
    output_path = resolve_output_path(
        args.output_dir,
        args.output_file,
        args.input,
        timestamp_output=args.timestamp_output,
    )

    df.to_csv(output_path, index=False)

    LOGGER.info("Saved raw data to %s", output_path)
    LOGGER.info("Rows: %s | Columns: %s", df.shape[0], df.shape[1])


if __name__ == "__main__":
    main()
