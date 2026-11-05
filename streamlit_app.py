"""AgroSynapse Dashboard - agricultural data quality, analysis and yield forecasting."""

from __future__ import annotations

import csv
import html
import io
import json
from datetime import datetime
import os
import re
from pathlib import Path
import subprocess
import sys
from typing import Any
import warnings
from textwrap import dedent

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pandas.errors import ParserError

st.set_page_config(
    page_title="AgroSynapse",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="auto",
)

SETTINGS_PATH = Path("data/processed/app_settings.json")
DEFAULT_SETTINGS: dict[str, Any] = {
    # The committed demo source. kobo_export_raw.csv is the pipeline's ingest copy and is
    # gitignored, so it only exists after a local pipeline run.
    "raw_data_path": "data/raw/africa_crop_yield.csv",
    "raw_fallback_path": "data/raw/kobo_export_raw.csv",
    "clean_data_path": "data/processed/clean_data.csv",
    "features_data_path": "data/processed/features.csv",
    "metrics_path": "data/processed/model_metrics.json",
    "predictions_path": "data/processed/predictions.csv",
    "model_path": "data/processed/model_pipeline.joblib",
    "dashboard_hist_bins": 18,
    "top_regions_limit": 10,
    "enable_animations": True,
    "show_data_diff": True,
}
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SIDEBAR_LOGO_PATH = ASSETS_DIR / "agrosynapse-sidebar-logo.svg"
WORDMARK_LOGO_PATH = ASSETS_DIR / "agrosynapse-logo-wordmark.svg"
_PROJECT_ROOT = Path(__file__).resolve().parent

if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from analysis import (  # noqa: E402
    benchmark_against_peers,
    compare_groups,
    format_p,
    group_difference_test,
    iita_crop_group,
    measure_label,
    measure_unit,
    period_means,
    period_values,
    quality_report,
    trend_by_group,
    write_summary,
)
from feature_config import is_risky_column  # noqa: E402


def _safe_data_path(path_str: str) -> Path:
    """Resolve a user-supplied path and verify it stays within the project directory.

    Raises ValueError for empty strings or paths that escape the project root.
    """
    raw = path_str.strip()
    if not raw:
        raise ValueError("Path must not be empty.")
    resolved = Path(raw).expanduser().resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT)
    except ValueError:
        raise ValueError(
            f"Path '{raw}' resolves outside the project directory and cannot be used."
        )
    return resolved


def load_app_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return DEFAULT_SETTINGS.copy()
    try:
        loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return DEFAULT_SETTINGS.copy()
    merged = DEFAULT_SETTINGS.copy()
    if isinstance(loaded, dict):
        merged.update(loaded)
    return merged


def save_app_settings(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


if "app_settings" not in st.session_state:
    st.session_state["app_settings"] = load_app_settings()


# -----------------------------------------------------------------------------
# Deployment mode
# -----------------------------------------------------------------------------
# Streamlit serves every visitor from one container sharing one filesystem, so any
# visitor who retrains the model or edits settings changes what everyone else sees. Demo
# mode is therefore the default: read-only unless the operator opts in.
#
#   Local development:  set AGROSYNAPSE_MODE=full
#   Hosted, with admin:  .streamlit/secrets.toml ->  [agrosynapse]
#                                                    admin_password = "..."
DEMO_NOTICE = (
    "This is a shared demo. You can upload a CSV to explore it - it stays in your browser "
    "session and is never saved. Saving files, retraining the model and changing settings "
    "are turned off because every visitor shares one server."
)


def _secret(key: str) -> str | None:
    try:
        return str(st.secrets["agrosynapse"][key])
    except Exception:
        return None


def admin_password() -> str | None:
    return _secret("admin_password") or os.environ.get("AGROSYNAPSE_ADMIN_PASSWORD") or None


def is_demo_mode() -> bool:
    """True when shared-state writes (uploads, retraining, settings) are blocked."""
    if st.session_state.get("admin_unlocked"):
        return False
    mode = (_secret("mode") or os.environ.get("AGROSYNAPSE_MODE") or "demo").strip().lower()
    return mode != "full"


def render_admin_unlock() -> None:
    """Offer a password box so the operator can write on a hosted deployment."""
    password = admin_password()
    if not password or not is_demo_mode():
        return
    with st.sidebar.expander("Operator access", icon=":material/admin_panel_settings:"):
        with st.form("admin_unlock_form", border=False):
            entered = st.text_input("Admin password", type="password", key="admin_password_input")
            unlock = st.form_submit_button("Unlock write access", key="admin_unlock_button", width="stretch")
        if unlock:
            if entered == password:
                st.session_state["admin_unlocked"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")


def flash(message: str) -> None:
    """Rerun, then confirm with a toast; a toast raised right before a rerun is lost."""
    st.session_state["flash_message"] = message
    st.rerun()


def guard_write_action(action: str) -> bool:
    """Return True when a state-changing action may proceed; otherwise explain why not."""
    if not is_demo_mode():
        return True
    st.warning(f"{action} is disabled in demo mode. {DEMO_NOTICE}")
    return False


def parse_hex_color(color_value: str | None) -> tuple[int, int, int] | None:
    if not color_value:
        return None
    color = color_value.strip()
    if color.startswith("#") and len(color) == 7:
        try:
            return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
        except ValueError:
            return None
    if color.startswith("#") and len(color) == 4:
        try:
            return (
                int(color[1] * 2, 16),
                int(color[2] * 2, 16),
                int(color[3] * 2, 16),
            )
        except ValueError:
            return None
    return None


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255.0


def theme_palette() -> dict[str, Any]:
    # This dashboard now uses a fixed dark theme via Streamlit config to
    # avoid conflicting live theme overrides from multiple custom CSS layers.
    base_opt = (st.get_option("theme.base") or "dark").lower()
    is_dark = base_opt != "light"

    bg_opt = st.get_option("theme.backgroundColor")
    bg_rgb = parse_hex_color(bg_opt)
    if bg_rgb is not None:
        is_dark = relative_luminance(bg_rgb) < 0.54

    if is_dark:
        return {
            "is_dark": True,
            "bg": bg_opt or "#0b121c",
            "secondary_bg": st.get_option("theme.secondaryBackgroundColor") or "#101a24",
            "text": st.get_option("theme.textColor") or "#e7ece9",
            "muted": "#95a39b",
            "line": "rgba(127, 207, 159, 0.28)",
            "donut_track": "rgba(58,72,92,0.45)",
        }
    return {
        "is_dark": False,
        "bg": bg_opt or "#f8fbf9",
        "secondary_bg": st.get_option("theme.secondaryBackgroundColor") or "#f3f7f4",
        "text": st.get_option("theme.textColor") or "#24312d",
        "muted": "#5a6a62",
        "line": "rgba(47, 106, 74, 0.22)",
        "donut_track": "rgba(118,139,126,0.42)",
    }


THEME = theme_palette()
IS_DARK_THEME = bool(THEME["is_dark"])

# Categorical palette for crop legends and donut segments. Defined once so the
# legend chips and the arc they describe cannot drift apart.
CHART_PALETTE: list[str] = [
    "#52d273",
    "#34c2a6",
    "#49a9f1",
    "#8479ff",
    "#d26cf3",
    "#f39a58",
    "#f0627a",
    "#b6d657",
]

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
# The stylesheet in assets/theme_static.css is the single source of truth for
# every visual token. Values that depend on runtime state are injected below as
# custom-property overrides rather than as a second, competing stylesheet.
_STATIC_CSS_PATH = ASSETS_DIR / "theme_static.css"
if _STATIC_CSS_PATH.exists():
    st.markdown(
        f"<style>{_STATIC_CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def apply_theme_attribute(is_dark: bool) -> None:
    """Stamp data-theme on the document root so the stylesheet can switch themes.

    Streamlit exposes the configured theme to Python but not to CSS, and it
    strips <script> from st.markdown. st.html renders inline and can run a
    script, so the attribute lands on the real document root.
    """
    theme_name = "dark" if is_dark else "light"
    st.html(
        f'<script>document.documentElement.setAttribute("data-theme", "{theme_name}");</script>',
        unsafe_allow_javascript=True,
    )


apply_theme_attribute(IS_DARK_THEME)

if not st.session_state.get("app_settings", DEFAULT_SETTINGS).get("enable_animations", True):
    # Opt-out for the "enable animations" setting. The stylesheet also honours
    # prefers-reduced-motion on its own, for users who ask the OS instead.
    st.markdown(
        """
        <style>
        .kpi-card, .legend-chip {
            animation: none !important;
            opacity: 1 !important;
            transform: none !important;
        }
        .gauge-seg {
            animation: none !important;
            stroke-dasharray: var(--len) 100 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def parse_csv_bytes(data: bytes) -> pd.DataFrame | None:
    """Parse CSV bytes, detecting encoding and delimiter. Returns None if unreadable."""
    sample_bytes = data[:4096]
    detected_encoding = "utf-8"
    sample = ""
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            sample = sample_bytes.decode(enc)
            detected_encoding = enc
            break
        except UnicodeDecodeError:
            # A multi-byte character cut off at the 4096-byte boundary also lands here;
            # latin1 accepts any byte, so the loop always ends with a usable sample.
            continue

    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;	|").delimiter
    except csv.Error:
        sep = ","

    try:
        return pd.read_csv(io.BytesIO(data), encoding=detected_encoding, sep=sep)
    except ParserError:
        fallback_sep = ";" if sep != ";" else ","
        try:
            return pd.read_csv(io.BytesIO(data), encoding=detected_encoding, sep=fallback_sep)
        except (ParserError, UnicodeDecodeError, ValueError):
            return None
    except (UnicodeDecodeError, ValueError):
        # EmptyDataError is a ValueError subclass: a zero-byte or header-less file.
        return None


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        return parse_csv_bytes(file_path.read_bytes())
    except OSError:
        return None


@st.cache_data(show_spinner=False)
def load_metrics(path: str) -> dict | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_modified_label(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "-"


def file_size_label(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "-"
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def settings_path_health(label: str, path_value: str, kind: str) -> dict[str, str]:
    raw_path = str(path_value).strip()
    if not raw_path:
        return {
            "Setting": label,
            "Status": "Missing",
            "Path": "-",
            "Details": "No path configured.",
            "Updated": "-",
        }

    path = Path(raw_path).expanduser()
    if not path.exists():
        return {
            "Setting": label,
            "Status": "Missing",
            "Path": raw_path,
            "Details": "File not found.",
            "Updated": "-",
        }

    details = f"{file_size_label(path)} file"
    status = "Ready"

    if kind == "csv":
        try:
            df = load_csv(str(path))
            if df is None:
                status = "Problem"
                details = "CSV could not be loaded."
            elif df.empty:
                status = "Problem"
                details = "CSV is empty."
            else:
                details = f"{format_num(df.shape[0], 0)} rows, {format_num(df.shape[1], 0)} fields"
        except Exception as exc:  # pragma: no cover - UI guard for unknown user files
            status = "Problem"
            details = f"CSV parse failed: {exc}"
    elif kind == "json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                details = f"{len(payload)} metric entries"
            else:
                details = "JSON loaded."
        except (json.JSONDecodeError, OSError) as exc:
            status = "Problem"
            details = f"JSON read failed: {exc}"
    elif kind == "model":
        details = f"Model artifact, {file_size_label(path)}"

    return {
        "Setting": label,
        "Status": status,
        "Path": raw_path,
        "Details": details,
        "Updated": file_modified_label(path),
    }


def settings_path_health_rows(settings: dict[str, Any]) -> list[dict[str, str]]:
    return [
        settings_path_health("Primary raw CSV", str(settings["raw_data_path"]), "csv"),
        settings_path_health("Fallback raw CSV", str(settings["raw_fallback_path"]), "csv"),
        settings_path_health("Clean CSV", str(settings["clean_data_path"]), "csv"),
        settings_path_health("Features CSV", str(settings["features_data_path"]), "csv"),
        settings_path_health("Metrics JSON", str(settings["metrics_path"]), "json"),
        settings_path_health("Predictions CSV", str(settings["predictions_path"]), "csv"),
        settings_path_health("Model artifact", str(settings["model_path"]), "model"),
    ]


def settings_readiness_summary(settings: dict[str, Any]) -> tuple[int, str, list[str]]:
    primary_raw = Path(str(settings["raw_data_path"])).expanduser().exists()
    fallback_raw = Path(str(settings["raw_fallback_path"])).expanduser().exists()
    clean_ready = Path(str(settings["clean_data_path"])).expanduser().exists()
    features_ready = Path(str(settings["features_data_path"])).expanduser().exists()
    metrics_ready = Path(str(settings["metrics_path"])).expanduser().exists()
    predictions_ready = Path(str(settings["predictions_path"])).expanduser().exists()
    model_ready = Path(str(settings["model_path"])).expanduser().exists()

    score = 0
    score += 30 if primary_raw or fallback_raw else 0
    score += 20 if clean_ready else 0
    score += 15 if features_ready else 0
    score += 15 if metrics_ready else 0
    score += 10 if predictions_ready else 0
    score += 10 if model_ready else 0

    if score >= 85:
        label = "Ready"
    elif score >= 55:
        label = "Needs attention"
    else:
        label = "Setup incomplete"

    recommendations: list[str] = []
    if not (primary_raw or fallback_raw):
        recommendations.append("Upload a CSV from the Uploads page or set a valid raw data path.")
    elif not primary_raw and fallback_raw:
        recommendations.append("Primary raw CSV is missing, but the fallback raw CSV exists.")
    if not clean_ready or not features_ready:
        recommendations.append("Run the pipeline after choosing the source CSV.")
    if not metrics_ready or not model_ready:
        recommendations.append("Model outputs are incomplete; training may have been skipped or not run yet.")
    if not predictions_ready and model_ready:
        recommendations.append("Model artifact exists, but predictions are missing.")
    if not recommendations:
        recommendations.append("Settings look ready for dashboard use.")

    return score, label, recommendations


def save_uploaded_csv(uploaded_file: Any, destination_path: str) -> tuple[bool, str]:
    if uploaded_file is None:
        return False, "No CSV selected."

    try:
        target = _safe_data_path(destination_path)
    except ValueError as exc:
        return False, str(exc)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(uploaded_file.getbuffer()))
    except OSError as exc:
        return False, f"Could not save CSV to `{target}`: {exc}"

    # Invalidate cache so the validation read and all subsequent reads are fresh.
    load_csv.clear()
    parsed = load_csv(str(target))
    if parsed is None:
        return False, f"Saved file at `{target}`, but it could not be parsed as CSV."
    if parsed.empty:
        return False, f"Saved file at `{target}`, but it has no rows."

    # Deliberately no cleanup pass here. This used to delete every other *.csv under
    # data/ after a successful upload, which took the committed demo dataset and the
    # pipeline outputs (clean_data.csv, predictions.csv, target.csv) with it. Each
    # pipeline step already overwrites the file it owns, so stale CSVs are not a problem
    # worth destroying data to solve.
    return True, f"Saved `{target}` ({parsed.shape[0]} rows, {parsed.shape[1]} columns)."


# -----------------------------------------------------------------------------
# Per-visitor uploads
# -----------------------------------------------------------------------------
# An uploaded CSV is parsed straight into st.session_state and never written to disk.
# Session state belongs to one browser session, so on a shared deployment each visitor
# explores their own file, nobody else can see it, and it is gone when the tab closes.
UPLOAD_SESSION_KEY = "uploaded_dataset"
MAX_UPLOAD_ROWS = 250_000
_FILTER_STATE_KEYS = (
    "global_district_filter",
    "global_crop_filter",
    "global_district_filter_choice",
    "global_crop_filter_choice",
)


def session_upload() -> dict[str, Any] | None:
    upload = st.session_state.get(UPLOAD_SESSION_KEY)
    return upload if isinstance(upload, dict) else None


def _reset_filter_state() -> None:
    # Filter selections from one dataset rarely exist in the next.
    for key in _FILTER_STATE_KEYS:
        st.session_state.pop(key, None)


def ingest_session_upload(uploaded_file: Any) -> str | None:
    """Parse an uploaded CSV into this session. Returns an error message, or None."""
    upload_id = getattr(uploaded_file, "file_id", None) or f"{uploaded_file.name}:{uploaded_file.size}"
    current = session_upload()
    if current and current.get("upload_id") == upload_id:
        return None

    raw = parse_csv_bytes(uploaded_file.getvalue())
    if raw is None:
        return f"`{uploaded_file.name}` could not be read as a CSV file."
    if raw.empty or raw.shape[1] == 0:
        return f"`{uploaded_file.name}` has no data rows."
    if len(raw) > MAX_UPLOAD_ROWS:
        return (
            f"`{uploaded_file.name}` has {len(raw):,} rows; this demo explores files of up to "
            f"{MAX_UPLOAD_ROWS:,} rows. Run the app locally for larger datasets."
        )

    # Same column-name normalisation the pipeline's clean step applies, so the dashboard
    # sees an upload the way it sees the pipeline output. Duplicates get a suffix.
    seen: dict[str, int] = {}
    columns: list[str] = []
    for column in raw.columns:
        base = normalize_column_name(str(column))
        seen[base] = seen.get(base, -1) + 1
        columns.append(base if seen[base] == 0 else f"{base}_{seen[base]}")

    st.session_state[UPLOAD_SESSION_KEY] = {
        "upload_id": upload_id,
        "name": uploaded_file.name,
        "raw": raw,
        "frame": raw.set_axis(columns, axis=1),
    }
    _reset_filter_state()
    return None


def upload_widget_key() -> str:
    return f"dataset_upload_{st.session_state.get('upload_widget_nonce', 0)}"


def clear_session_upload() -> None:
    st.session_state.pop(UPLOAD_SESSION_KEY, None)
    # A file_uploader cannot be emptied programmatically, and one still holding the file
    # would re-ingest it on the next run. A fresh key renders a fresh, empty widget.
    st.session_state["upload_widget_nonce"] = st.session_state.get("upload_widget_nonce", 0) + 1
    _reset_filter_state()


def upload_privacy_note(upload: dict[str, Any]) -> str:
    frame = upload["frame"]
    return (
        f"Exploring your upload **{upload['name']}** "
        f"({frame.shape[0]:,} rows, {frame.shape[1]} columns). It lives only in this browser "
        "session: it is not saved on the server and no other visitor can see it."
    )


def run_pipeline_for_input(input_path: str, target_column: str, settings: dict[str, Any]) -> tuple[bool, str]:
    try:
        input_file = _safe_data_path(input_path)
    except ValueError as exc:
        return False, str(exc)
    if not input_file.exists():
        return False, f"Input CSV does not exist: `{input_file}`"

    features_output_dir = Path(str(settings["features_data_path"])).expanduser().parent
    command = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--input",
        str(input_file),
        "--target",
        target_column.strip() or "auto",
        "--ingested-output",
        str(Path(str(settings["raw_data_path"])).expanduser()),
        "--clean-output",
        str(Path(str(settings["clean_data_path"])).expanduser()),
        "--features-output-dir",
        str(features_output_dir),
        "--model-output",
        str(Path(str(settings["model_path"])).expanduser()),
        "--metrics-output",
        str(Path(str(settings["metrics_path"])).expanduser()),
        "--predictions-output",
        str(Path(str(settings["predictions_path"])).expanduser()),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output_chunks = [chunk.strip() for chunk in [completed.stdout, completed.stderr] if chunk and chunk.strip()]
    output_text = "\n\n".join(output_chunks)
    if completed.returncode != 0:
        return False, output_text or "Pipeline failed with a non-zero exit code."
    return True, output_text or "Pipeline completed successfully."


def target_health_message(
    df: pd.DataFrame | None,
    requested_target: str,
    inferred_target: str | None = None,
    auto_detect: bool = False,
) -> tuple[str, str]:
    if df is None or df.empty:
        return ("unknown", "Upload or save a CSV to evaluate target availability.")

    target_candidates = ["yield_tons", "yield", "yield_kg", "actual_yield_tons"]
    lower_map = {str(col).lower(): str(col) for col in df.columns}
    if auto_detect:
        if inferred_target and inferred_target in df.columns:
            return ("good", f"Auto-detected target: `{inferred_target}`.")
        return (
            "warn",
            "Auto-detect could not find a reliable target column. "
            "The pipeline will run data prep and may skip training.",
        )

    requested = requested_target.strip().lower()
    if requested and requested in lower_map:
        return ("good", f"Requested target found: `{lower_map[requested]}`.")

    if inferred_target and inferred_target in df.columns:
        return (
            "warn",
            f"Requested target not found. Suggested target: `{inferred_target}`.",
        )

    matched = [lower_map[name] for name in target_candidates if name in lower_map]
    if matched:
        return ("warn", f"Requested target not found. Fallback target candidate available: `{matched[0]}`.")
    return (
        "warn",
        f"Requested target `{requested_target}` not found, and no fallback target-like column was detected. "
        "Pipeline will run ingestion/cleaning/features and skip model training.",
    )


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def pick_geography_column(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None

    deny_tokens = {
        "full_name",
        "farmer_name",
        "name",
        "farmer_id",
        "phone",
        "phone_number",
        "id",
        "registration_date",
        "sample_date",
    }
    primary_tokens = ["district", "lga", "ward", "village", "community", "location"]
    secondary_tokens = ["state", "region"]

    best_col = None
    best_score = -10_000
    n_rows = max(len(df), 1)

    for col in df.columns:
        col_norm = normalize_column_name(col)
        if any(tok in col_norm for tok in deny_tokens):
            continue

        series = df[col].dropna()
        if series.empty:
            continue
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        s = series.astype(str).str.strip()
        unique_count = int(s.nunique())
        if unique_count <= 1:
            continue

        unique_ratio = unique_count / n_rows
        score = 0

        if any(tok in col_norm for tok in primary_tokens):
            score += 80
        elif any(tok in col_norm for tok in secondary_tokens):
            score += 50
        else:
            continue

        if unique_ratio > 0.9:
            score -= 60
        elif unique_ratio > 0.7:
            score -= 35
        elif unique_ratio < 0.15:
            score -= 10
        else:
            score += 15

        if any("," in val for val in s.head(15)):
            score -= 8

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def pretty_column_label(name: str) -> str:
    """Readable, sentence-case label with the unit pulled out: ``yield_kg_per_ha`` -> ``Yield (kg/ha)``."""
    if not str(name).strip("_ "):
        return "Field"
    return axis_title(str(name))


def numeric_valid_ratio(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    series = pd.to_numeric(df[column], errors="coerce")
    return float(series.notna().mean())


def candidate_numeric_columns(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
    min_valid_ratio: float = 0.25,
) -> list[str]:
    if df is None or df.empty:
        return []

    excluded = set(exclude or [])
    scored: list[tuple[float, int, int, str]] = []
    n_rows = max(len(df), 1)

    for col in df.columns:
        if col in excluded:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid = int(series.notna().sum())
        if valid == 0:
            continue
        ratio = valid / n_rows
        if ratio < min_valid_ratio:
            continue
        unique = int(series.nunique(dropna=True))
        scored.append((ratio, unique, valid, col))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in scored]


def pick_numeric_column(
    df: pd.DataFrame,
    preferred: list[str] | None = None,
    exclude: list[str] | None = None,
    min_valid_ratio: float = 0.25,
) -> str | None:
    if df is None or df.empty:
        return None

    excluded = set(exclude or [])
    lower_map = {str(col).lower(): str(col) for col in df.columns if col not in excluded}

    for name in preferred or []:
        found = lower_map.get(str(name).lower())
        if found and numeric_valid_ratio(df, found) >= min_valid_ratio:
            return found

    candidates = candidate_numeric_columns(df, exclude=list(excluded), min_valid_ratio=min_valid_ratio)
    return candidates[0] if candidates else None


def pick_category_column(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
    preferred_tokens: list[str] | None = None,
) -> str | None:
    if df is None or df.empty:
        return None

    excluded = set(exclude or [])
    tokens = preferred_tokens or []
    n_rows = max(len(df), 1)
    best_col = None
    best_score = -10_000.0

    for col in df.columns:
        if col in excluded:
            continue

        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        ratio = float(len(non_null)) / float(n_rows)
        if ratio < 0.3:
            continue

        as_text = non_null.astype(str).str.strip()
        as_text = as_text[as_text != ""]
        if as_text.empty:
            continue

        unique = int(as_text.nunique())
        if unique < 2:
            continue
        max_unique = max(8, min(60, int(n_rows * 0.45)))
        if unique > max_unique:
            continue

        col_norm = normalize_column_name(str(col))
        score = (ratio * 100.0) - (unique * 0.5)
        if any(token in col_norm for token in tokens):
            score += 20.0
        if not pd.api.types.is_numeric_dtype(series):
            score += 6.0

        if score > best_score:
            best_score = score
            best_col = str(col)

    return best_col


def infer_target_column(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None

    priority = [
        "yield_kg_per_ha",
        "yield_tons",
        "yield",
        "yield_kg",
        "actual_yield_tons",
        "target",
        "price",
        "usdprice",
        "value",
        "amount",
        "cost",
        "sales",
        "revenue",
        "score",
    ]

    direct = pick_column(df, priority)
    if direct is not None:
        return direct

    candidates = candidate_numeric_columns(df, min_valid_ratio=0.45)
    if not candidates:
        return None

    deprioritize_tokens = ["lat", "lon", "latitude", "longitude", "year", "month", "day", "id", "index"]

    def score(col: str) -> float:
        norm = normalize_column_name(col)
        ratio = numeric_valid_ratio(df, col)
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        unique = float(numeric_series.nunique(dropna=True))
        token_bonus = 28.0 if any(tok in norm for tok in priority) else 0.0
        token_penalty = -20.0 if any(tok in norm for tok in deprioritize_tokens) else 0.0
        return (ratio * 100.0) + min(unique, 1000.0) / 25.0 + token_bonus + token_penalty

    return max(candidates, key=score)


def normalize_column_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized or "field"


def format_num(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, str):
        stripped = value.strip()
        try:
            numeric_value = float(stripped)
        except ValueError:
            return stripped
    else:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)
    if abs(numeric_value) >= 1000:
        return f"{numeric_value:,.0f}"
    return f"{numeric_value:.{digits}f}"


def sanitize_block(markup: str) -> str:
    return dedent(markup).strip()


def render_html_block(markup: str) -> None:
    st.markdown(sanitize_block(markup), unsafe_allow_html=True)


def render_markdown_block(markup: str) -> None:
    st.markdown(sanitize_block(markup))


def kpi_card(label: str, value: str, delta: str = "", icon: str = "", color: str = "") -> None:
    accent = color or "#1f6b3b"
    accent_soft = hex_to_rgba(accent, 0.12) or "rgba(31, 107, 59, 0.10)"
    accent_border = hex_to_rgba(accent, 0.22) or "rgba(31, 107, 59, 0.16)"
    card_class = "kpi-card" if icon else "kpi-card no-icon"
    top_class = "kpi-card-top" if icon else "kpi-card-top no-icon"
    icon_html = (
        f'<div class="kpi-icon" style="background:{accent_soft}; border-color:{accent_border}; color:{accent};">{icon}</div>'
        if icon
        else ""
    )
    delta_class = "kpi-delta" if delta else "kpi-delta kpi-delta-empty"
    delta_html = html.escape(delta) if delta else "&nbsp;"
    label_html = f'<div class="kpi-label">{html.escape(label)}</div>'
    value_html = f'<div class="kpi-value">{html.escape(value)}</div>'

    card_html = (
        f'<div class="{card_class}" style="--accent:{accent};">'
        f'<div class="{top_class}">{label_html}{icon_html}</div>'
        f'{value_html}'
        f'<div class="{delta_class}">{delta_html}</div>'
        '</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)

def hex_to_rgba(color_value: str | None, alpha: float = 0.14) -> str | None:
    rgb = parse_hex_color(color_value)
    if rgb is None:
        return None
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"


_CHIP_OK = {"available", "ready", "ok", "passed", "complete", "enabled", "on", "yes"}
_CHIP_BAD = {"missing", "failed", "error", "none", "unavailable", "no"}
_CHIP_WARN = {"stale", "partial", "pending", "unknown", "never"}


def chip_tone(value: object) -> str:
    """Map a chip value to a status class so state reads at a glance."""
    text = str(value).strip().lower()
    if text in _CHIP_OK:
        return "is-ok"
    if text in _CHIP_BAD:
        return "is-danger"
    if text in _CHIP_WARN:
        return "is-warn"
    return ""


def render_page_header(
    title: str,
    subtitle: str = "",
    chips: list[tuple[str, str]] | None = None,
    eyebrow: str = "",
) -> None:
    """Render a page header.

    Every argument is rendered. Callers pass status chips carrying real state
    (dataset availability, last pipeline run), so dropping them silently would
    hide information the page has already computed.
    """
    parts: list[str] = []

    if eyebrow:
        parts.append(f'<div class="page-header-eyebrow">{html.escape(eyebrow)}</div>')

    parts.append(
        '<div class="page-topbar">'
        f'<h1 class="page-topbar-title">{html.escape(title)}</h1>'
        '<div class="page-topbar-rule"></div>'
        '</div>'
    )

    if subtitle:
        parts.append(f'<p class="page-header-subtitle">{html.escape(subtitle)}</p>')

    if chips:
        chip_html = "".join(
            f'<span class="page-chip {chip_tone(value)}">'
            f'<span class="page-chip-label">{html.escape(str(label))}</span>'
            f'<span class="page-chip-value">{html.escape(str(value))}</span>'
            '</span>'
            for label, value in chips
        )
        parts.append(f'<div class="page-chips">{chip_html}</div>')

    st.markdown("".join(parts), unsafe_allow_html=True)


def insight_card_html(
    title: str,
    body: str,
    kicker: str = "",
    tone: str = "neutral",
    footnote: str = "",
) -> str:
    tone_class = tone if tone in {"positive", "warning", "neutral", "accent"} else "neutral"
    kicker_html = f'<div class="insight-kicker">{html.escape(kicker)}</div>' if kicker else ""
    footnote_html = f'<div class="insight-footnote">{html.escape(footnote)}</div>' if footnote else ""
    return (
        f'<div class="smart-insight-card {tone_class}">'
        f'{kicker_html}'
        f'<div class="insight-title">{html.escape(title)}</div>'
        f'<div class="insight-body">{html.escape(body)}</div>'
        f'{footnote_html}'
        '</div>'
    )


def render_insight_cards(insights: list[dict[str, str]], per_row: int = 2) -> None:
    """Lay the cards out in one CSS grid, so cards in a row share a height and stack on phones."""
    cleaned = [item for item in insights if item]
    if not cleaned:
        return
    cards = "".join(
        insight_card_html(
            title=item.get("title", ""),
            body=item.get("body", ""),
            kicker=item.get("kicker", ""),
            tone=item.get("tone", "neutral"),
            footnote=item.get("footnote", ""),
        )
        for item in cleaned
    )
    st.markdown(
        f'<div class="insight-grid" style="--cols:{max(1, per_row)};">{cards}</div>',
        unsafe_allow_html=True,
    )


def datetime_valid_ratio(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0

    series = df[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        return float(series.notna().mean())

    col_norm = normalize_column_name(str(column))
    if pd.api.types.is_numeric_dtype(series) and not any(
        token in col_norm for token in ["date", "time", "year", "month", "week", "day", "season"]
    ):
        return 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format")
        parsed = pd.to_datetime(series, errors="coerce")

    return float(parsed.notna().mean())


def parse_datetime_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="datetime64[ns]")

    series = df[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format")
        return pd.to_datetime(series, errors="coerce")


def pick_datetime_column(
    df: pd.DataFrame,
    preferred: list[str] | None = None,
    min_valid_ratio: float = 0.32,
) -> str | None:
    if df is None or df.empty:
        return None

    lower_map = {str(col).lower(): str(col) for col in df.columns}
    for name in preferred or [
        "date",
        "survey_date",
        "collection_date",
        "registration_date",
        "sample_date",
        "created_at",
        "timestamp",
        "season",
    ]:
        found = lower_map.get(str(name).lower())
        if found and datetime_valid_ratio(df, found) >= min_valid_ratio:
            return found

    best_col = None
    best_score = -10_000.0
    for col in df.columns:
        ratio = datetime_valid_ratio(df, str(col))
        if ratio < min_valid_ratio:
            continue

        norm = normalize_column_name(str(col))
        score = ratio * 100.0
        if any(token in norm for token in ["date", "time", "year", "month", "week", "day", "season"]):
            score += 24.0
        if pd.api.types.is_numeric_dtype(df[col]):
            score -= 12.0

        unique_count = int(df[col].nunique(dropna=True))
        if unique_count <= 1:
            continue
        if unique_count > max(10, int(len(df) * 0.95)):
            score -= 6.0

        if score > best_score:
            best_score = score
            best_col = str(col)

    return best_col


def format_period_label(period: pd.Period, granularity: str) -> str:
    if granularity == "year":
        return str(period)
    if granularity == "month":
        return period.strftime("%b %Y")
    if granularity == "week":
        return f"Week of {period.start_time.strftime('%d %b %Y')}"
    return period.strftime("%d %b %Y")


def time_trend_insight(
    df: pd.DataFrame,
    metric_col: str | None,
    date_col: str | None = None,
) -> dict[str, str] | None:
    if df is None or df.empty or metric_col is None:
        return None

    resolved_date = date_col or pick_datetime_column(df)
    if resolved_date is None:
        return None

    working = pd.DataFrame(
        {
            "__date": parse_datetime_series(df, resolved_date),
            "__metric": pd.to_numeric(df[metric_col], errors="coerce"),
        }
    ).dropna()

    if len(working) < 8:
        return None

    span_days = int((working["__date"].max() - working["__date"].min()).days)
    if span_days >= 540:
        granularity = "year"
        grouped = working.groupby(working["__date"].dt.to_period("Y"))["__metric"].mean().sort_index()
    elif span_days >= 120:
        granularity = "month"
        grouped = working.groupby(working["__date"].dt.to_period("M"))["__metric"].mean().sort_index()
    elif span_days >= 35:
        granularity = "week"
        grouped = working.groupby(working["__date"].dt.to_period("W"))["__metric"].mean().sort_index()
    else:
        granularity = "day"
        grouped = working.groupby(working["__date"].dt.to_period("D"))["__metric"].mean().sort_index()

    if len(grouped) < 2:
        return None

    latest_period = grouped.index[-1]
    previous_period = grouped.index[-2]
    latest_value = float(grouped.iloc[-1])
    previous_value = float(grouped.iloc[-2])
    delta_value = latest_value - previous_value

    if abs(previous_value) < 1e-9:
        delta_pct = None
    else:
        delta_pct = (delta_value / abs(previous_value)) * 100.0

    if delta_value > 0:
        direction_word = "higher"
        tone = "positive"
    elif delta_value < 0:
        direction_word = "lower"
        tone = "warning"
    else:
        direction_word = "flat"
        tone = "neutral"

    delta_phrase = ""
    if delta_pct is not None and delta_value != 0:
        delta_phrase = f" ({abs(delta_pct):.1f}% {'up' if delta_value > 0 else 'down'})"

    return {
        "kicker": "Trend signal",
        "title": f"{pretty_column_label(metric_col)} moved {direction_word}",
        "body": (
            f"{format_period_label(latest_period, granularity)} averaged {format_num(latest_value, 2)} "
            f"versus {format_num(previous_value, 2)} in {format_period_label(previous_period, granularity)}"
            f"{delta_phrase}."
        ),
        "tone": tone,
        "footnote": f"Time axis detected from {pretty_column_label(resolved_date)}.",
    }


def benchmark_gap_insight(
    df: pd.DataFrame,
    metric_col: str | None,
    group_col: str | None,
) -> dict[str, str] | None:
    if df is None or df.empty or metric_col is None or group_col is None or group_col == metric_col:
        return None
    if group_col not in df.columns:
        return None

    working = pd.DataFrame(
        {
            "__group": df[group_col].astype(str).str.strip(),
            "__metric": pd.to_numeric(df[metric_col], errors="coerce"),
        }
    )
    working = working.dropna()
    working = working[working["__group"] != ""]
    if len(working) < 10:
        return None

    min_group_size = 3 if len(working) >= 20 else 2
    grouped = (
        working.groupby("__group")["__metric"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    grouped = grouped[grouped["count"] >= min_group_size]
    if len(grouped) < 2:
        return None

    best_name = str(grouped.index[0])
    worst_name = str(grouped.index[-1])
    best_mean = float(grouped.iloc[0]["mean"])
    worst_mean = float(grouped.iloc[-1]["mean"])
    gap_value = best_mean - worst_mean
    gap_pct = None if abs(worst_mean) < 1e-9 else (gap_value / abs(worst_mean)) * 100.0
    pct_phrase = f" ({gap_pct:.1f}% spread)" if gap_pct is not None else ""

    return {
        "kicker": "Benchmark gap",
        "title": f"{best_name} leads on {pretty_column_label(metric_col)}",
        "body": (
            f"Across comparable {pretty_column_label(group_col).lower()} groups, {best_name} averages "
            f"{format_num(best_mean, 2)} while {worst_name} sits at {format_num(worst_mean, 2)}. "
            f"The current gap is {format_num(gap_value, 2)}{pct_phrase}."
        ),
        "tone": "positive",
        "footnote": f"Groups shown only when they have at least {min_group_size} records.",
    }


def practice_uplift_insight(
    df: pd.DataFrame,
    metric_col: str | None,
) -> dict[str, str] | None:
    if df is None or df.empty or metric_col is None:
        return None

    yes_values = {"yes", "true", "1", "y"}
    no_values = {"no", "false", "0", "n"}
    best_signal: tuple[float, dict[str, str]] | None = None

    for column_name, display_label, _ in detect_adoption_metrics(df):
        series = df[column_name].astype(str).str.strip().str.lower()
        yes_mask = series.isin(yes_values)
        no_mask = series.isin(no_values)
        valid_mask = yes_mask | no_mask
        if int(valid_mask.sum()) < 8:
            continue

        metric_values = pd.to_numeric(df.loc[valid_mask, metric_col], errors="coerce")
        valid_frame = pd.DataFrame(
            {
                "__metric": metric_values,
                "__adopter": yes_mask.loc[valid_mask].astype(bool).values,
            }
        ).dropna()
        if valid_frame.empty:
            continue

        adopters = valid_frame.loc[valid_frame["__adopter"], "__metric"]
        non_adopters = valid_frame.loc[~valid_frame["__adopter"], "__metric"]
        if len(adopters) < 3 or len(non_adopters) < 3:
            continue

        adopter_mean = float(adopters.mean())
        non_adopter_mean = float(non_adopters.mean())
        delta_value = adopter_mean - non_adopter_mean
        if abs(non_adopter_mean) < 1e-9:
            delta_pct = None
            score = abs(delta_value)
        else:
            delta_pct = (delta_value / abs(non_adopter_mean)) * 100.0
            score = abs(delta_pct)

        tone = "positive" if delta_value > 0 else "warning" if delta_value < 0 else "neutral"
        pct_phrase = f" ({delta_pct:+.1f}% vs non-users)" if delta_pct is not None else ""

        result = {
            "kicker": "Practice signal",
            "title": f"{display_label} links to a measurable shift",
            "body": (
                f"Records with {display_label.lower()} average {format_num(adopter_mean, 2)} versus "
                f"{format_num(non_adopter_mean, 2)} for non-users{pct_phrase}."
            ),
            "tone": tone,
            "footnote": "This is an observed association in the current slice, not proof of causation.",
        }
        if best_signal is None or score > best_signal[0]:
            best_signal = (score, result)

    return best_signal[1] if best_signal else None


def quality_or_coverage_insight(
    df: pd.DataFrame,
    metric_col: str | None = None,
    category_col: str | None = None,
) -> dict[str, str] | None:
    if df is None or df.empty:
        return None

    completeness = float(df.notna().sum().sum()) / float(max(df.shape[0] * max(df.shape[1], 1), 1)) * 100.0
    missing_rates = (1.0 - df.notna().mean()).sort_values(ascending=False)
    hotspots = missing_rates[missing_rates >= 0.25].head(3)

    if not hotspots.empty:
        hotspot_text = ", ".join(
            f"{pretty_column_label(col)} ({rate * 100:.0f}% missing)"
            for col, rate in hotspots.items()
        )
        return {
            "kicker": "Data risk",
            "title": "Missingness is concentrated in a few fields",
            "body": f"Overall completeness is {completeness:.1f}%. The biggest gaps are in {hotspot_text}.",
            "tone": "warning",
            "footnote": "Cleaning these fields first will improve chart reliability and model training.",
        }

    if metric_col is not None:
        numeric_series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
        if len(numeric_series) >= 12:
            q1 = float(numeric_series.quantile(0.25))
            q3 = float(numeric_series.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                high_outliers = int((numeric_series > (q3 + 1.5 * iqr)).sum())
                if high_outliers >= max(2, int(len(numeric_series) * 0.06)):
                    return {
                        "kicker": "Distribution signal",
                        "title": "A small set of high values is stretching the metric",
                        "body": (
                            f"{high_outliers} records sit well above the typical range for "
                            f"{pretty_column_label(metric_col)}. This usually means the average is being pulled upward."
                        ),
                        "tone": "neutral",
                        "footnote": "The dashboard caps only chart display for readability; the raw data stays unchanged.",
                    }

    if category_col is not None and category_col in df.columns:
        shares = df[category_col].dropna().astype(str).str.strip()
        if not shares.empty:
            normalized = shares[shares != ""].value_counts(normalize=True)
            if not normalized.empty and float(normalized.iloc[0]) >= 0.55:
                top_name = str(normalized.index[0])
                top_share = float(normalized.iloc[0]) * 100.0
                return {
                    "kicker": "Composition signal",
                    "title": f"{top_name} dominates the current slice",
                    "body": (
                        f"{top_name} accounts for {top_share:.1f}% of visible records. "
                        f"Interpret broad averages with that weighting in mind."
                    ),
                    "tone": "accent",
                    "footnote": f"Primary composition field: {pretty_column_label(category_col)}.",
                }

    tone = "positive" if completeness >= 85.0 else "neutral"
    return {
        "kicker": "Coverage check",
        "title": "The visible slice is structurally usable",
        "body": (
            f"Overall completeness is {completeness:.1f}%, which is strong enough for automated comparisons "
            "and narrative insights."
        ),
        "tone": tone,
        "footnote": "Signals become more dependable as completeness and record counts increase.",
    }


def dataset_profile_insight(df: pd.DataFrame, target_candidate: str | None = None) -> dict[str, str]:
    if df is None or df.empty:
        return {
            "kicker": "Schema profile",
            "title": "No dataset detected yet",
            "body": "Upload or point the app to a CSV to unlock automatic profiling.",
            "tone": "neutral",
            "footnote": "",
        }

    target = target_candidate or infer_target_column(df)
    geo_col = pick_geography_column(df)
    date_col = pick_datetime_column(df)
    category_col = pick_category_column(
        df,
        exclude=[target] if target else [],
        preferred_tokens=["crop", "type", "category", "indicator", "state", "region", "country", "district"],
    )
    adoption_metrics = detect_adoption_metrics(df)
    numeric_cols = candidate_numeric_columns(df, min_valid_ratio=0.25)

    pieces = [f"{len(df):,} records", f"{df.shape[1]:,} fields", f"{len(numeric_cols)} numeric measures"]
    if geo_col:
        pieces.append(f"geo field: {pretty_column_label(geo_col)}")
    if date_col:
        pieces.append(f"time field: {pretty_column_label(date_col)}")
    if category_col:
        pieces.append(f"category field: {pretty_column_label(category_col)}")
    if adoption_metrics:
        pieces.append(f"{len(adoption_metrics)} practice signal(s)")

    title = (
        f"{pretty_column_label(target)} is the strongest target candidate"
        if target
        else "The upload looks ready for exploratory analysis"
    )
    tone = "positive" if target else "neutral"
    footnote = (
        f"Detected target coverage: {numeric_valid_ratio(df, target) * 100:.0f}% numeric."
        if target
        else "A numeric outcome like Yield, Price, or Value will unlock model training."
    )

    return {
        "kicker": "Schema profile",
        "title": title,
        "body": "Detected " + ", ".join(pieces) + ".",
        "tone": tone,
        "footnote": footnote,
    }


def dataset_schema_overview(df: pd.DataFrame, target_candidate: str | None = None) -> dict[str, Any]:
    target = target_candidate or infer_target_column(df)
    geo_col = pick_geography_column(df)
    date_col = pick_datetime_column(df)
    category_col = pick_category_column(
        df,
        exclude=[target] if target else [],
        preferred_tokens=["crop", "type", "category", "indicator", "state", "region", "country", "district"],
    )
    adoption_metrics = detect_adoption_metrics(df)
    numeric_cols = candidate_numeric_columns(df, min_valid_ratio=0.25)
    completeness = float(df.notna().sum().sum()) / float(max(df.shape[0] * max(df.shape[1], 1), 1)) * 100.0

    signal_tokens: list[str] = []
    if geo_col:
        signal_tokens.append("Geo")
    if date_col:
        signal_tokens.append("Time")
    if category_col:
        signal_tokens.append("Category")
    if adoption_metrics:
        signal_tokens.append("Practices")

    return {
        "target": target,
        "geo_col": geo_col,
        "date_col": date_col,
        "category_col": category_col,
        "adoption_metrics": adoption_metrics,
        "adoption_count": len(adoption_metrics),
        "numeric_count": len(numeric_cols),
        "signals_count": len(signal_tokens),
        "signals_label": " • ".join(signal_tokens) if signal_tokens else "Core fields only",
        "completeness": completeness,
    }


def dataset_readiness_score(df: pd.DataFrame, target_candidate: str | None = None) -> tuple[int, str]:
    overview = dataset_schema_overview(df, target_candidate=target_candidate)
    score = 0
    score += int(min(40.0, round(overview["completeness"] * 0.4)))
    if overview["target"]:
        score += 20
    if overview["numeric_count"] >= 2:
        score += 10
    if overview["geo_col"]:
        score += 10
    if overview["date_col"]:
        score += 8
    if overview["category_col"]:
        score += 7
    if overview["adoption_count"] > 0:
        score += 5
    score = min(100, score)

    if score >= 85:
        band = "Model-ready"
    elif score >= 70:
        band = "Analysis-ready"
    elif score >= 55:
        band = "Good for prep"
    else:
        band = "Needs cleanup"

    return score, band


def build_dashboard_insights(
    df: pd.DataFrame,
    metric_col: str | None,
    comparison_candidates: list[str] | None = None,
    date_col: str | None = None,
    category_col: str | None = None,
) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []
    if df is None or df.empty:
        return insights

    if metric_col:
        trend = time_trend_insight(df, metric_col, date_col=date_col)
        if trend:
            insights.append(trend)

        comparison_col = next(
            (
                col
                for col in (comparison_candidates or [])
                if col and col in df.columns and col != metric_col
            ),
            None,
        )
        benchmark = benchmark_gap_insight(df, metric_col, comparison_col)
        if benchmark:
            insights.append(benchmark)

        practice = practice_uplift_insight(df, metric_col)
        if practice:
            insights.append(practice)

    quality = quality_or_coverage_insight(df, metric_col=metric_col, category_col=category_col)
    if quality:
        insights.append(quality)

    if not insights:
        insights.append(
            {
                "kicker": "Smart mode",
                "title": "The app is scanning this dataset adaptively",
                "body": "Add one solid numeric field and one grouping field to unlock richer auto-generated insights.",
                "tone": "neutral",
                "footnote": "Fields like yield, value, district, crop, and survey date work especially well.",
            }
        )

    deduped: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    for insight in insights:
        title = insight.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        deduped.append(insight)

    return deduped[:4]


def build_upload_preview_insights(
    df: pd.DataFrame,
    target_candidate: str | None = None,
) -> list[dict[str, str]]:
    if df is None or df.empty:
        return []

    overview = dataset_schema_overview(df, target_candidate=target_candidate)
    focus_metric = target_candidate or pick_numeric_column(
        df,
        preferred=["yield_tons", "yield", "value", "amount", "score", "price", "revenue"],
        min_valid_ratio=0.25,
    )
    comparison_candidates = [
        col
        for col in [overview["geo_col"], overview["category_col"]]
        if col
    ]

    insights: list[dict[str, str]] = [dataset_profile_insight(df, target_candidate=target_candidate)]

    if focus_metric:
        opportunity = (
            time_trend_insight(df, focus_metric, date_col=overview["date_col"])
            or benchmark_gap_insight(df, focus_metric, comparison_candidates[0] if comparison_candidates else None)
            or practice_uplift_insight(df, focus_metric)
        )
        if opportunity:
            opportunity["kicker"] = "Opportunity"
            insights.append(opportunity)

    quality = quality_or_coverage_insight(
        df,
        metric_col=focus_metric,
        category_col=overview["category_col"] or overview["geo_col"],
    )
    if quality:
        insights.append(quality)

    if not overview["target"]:
        insights.append(
            {
                "kicker": "Next step",
                "title": "Pick a numeric target before training",
                "body": (
                    "The pipeline can still clean and engineer features, but model training will be skipped unless you choose "
                    "a dependable outcome field such as Yield, Price, Value, or another numeric target."
                ),
                "tone": "warning",
                "footnote": "Use manual target override if the right field is present but not auto-detected.",
            }
        )

    deduped: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    for insight in insights:
        title = insight.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        deduped.append(insight)

    return deduped[:3]


def app_settings() -> dict[str, Any]:
    settings = st.session_state.get("app_settings")
    if not isinstance(settings, dict):
        settings = DEFAULT_SETTINGS.copy()
        st.session_state["app_settings"] = settings
    return settings


def get_active_frame() -> pd.DataFrame | None:
    upload = session_upload()
    if upload is not None:
        return upload["frame"]
    settings = app_settings()
    clean_df = load_csv(str(settings["clean_data_path"]))
    if clean_df is not None:
        return clean_df
    for raw_path in [str(settings["raw_data_path"]), str(settings["raw_fallback_path"])]:
        raw_df = load_csv(raw_path)
        if raw_df is not None:
            return raw_df
    return None


def pipeline_artifacts() -> list[tuple[str, Path]]:
    settings = app_settings()
    return [
        ("Raw", Path(str(settings["raw_data_path"]))),
        ("Clean", Path(str(settings["clean_data_path"]))),
        ("Model", Path(str(settings["model_path"]))),
        ("Metrics", Path(str(settings["metrics_path"]))),
        ("Predictions", Path(str(settings["predictions_path"]))),
    ]


def last_run_label() -> str:
    """When the pipeline last wrote an artifact, or "Never"."""
    existing_paths = [path for _, path in pipeline_artifacts() if path.exists()]
    if not existing_paths:
        return "Never"
    latest = max(existing_paths, key=lambda p: p.stat().st_mtime)
    return datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def data_diff_summary() -> dict[str, object]:
    settings = app_settings()
    raw_df = None
    for raw_path in [str(settings["raw_data_path"]), str(settings["raw_fallback_path"])]:
        raw_df = load_csv(raw_path)
        if raw_df is not None:
            break
    clean_df = load_csv(str(settings["clean_data_path"]))
    preds_df = load_csv(str(settings["predictions_path"]))

    if raw_df is None or clean_df is None:
        return {}

    raw_cols = list(raw_df.columns)
    clean_cols = list(clean_df.columns)
    norm_map = {c: normalize_column_name(c) for c in raw_cols}
    normalized_raw = set(norm_map.values())
    clean_set = set(clean_cols)

    renamed_pairs = [
        {"from": original, "to": normalized}
        for original, normalized in norm_map.items()
        if original != normalized and normalized in clean_set
    ]
    removed_after_clean = sorted(normalized_raw - clean_set)

    added_by_prediction: list[str] = []
    if preds_df is not None:
        added_by_prediction = sorted(set(preds_df.columns) - clean_set)

    return {
        "raw_shape": raw_df.shape,
        "clean_shape": clean_df.shape,
        "raw_cols": len(raw_cols),
        "clean_cols": len(clean_cols),
        "renamed_pairs": renamed_pairs,
        "removed_after_clean": removed_after_clean,
        "added_by_prediction": added_by_prediction,
    }


def data_contract_check(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if df is None or df.empty:
        return [
            {
                "Field": "Records",
                "Status": "Missing",
                "Mapped column": "0 rows",
                "Priority": "Required",
            }
        ]

    def add_row(field: str, found: bool, mapped: str, priority: str) -> None:
        rows.append(
            {
                "Field": field,
                "Status": "Found" if found else "Missing",
                "Mapped column": mapped if mapped else "-",
                "Priority": priority,
            }
        )

    add_row("Records", True, f"{len(df):,} rows", "Required")

    target_col = pick_column(
        df,
        ["yield_kg_per_ha", "yield_t_per_ha", "yield_tons", "yield", "yield_kg", "actual_yield_tons", "value", "target"],
    )
    add_row("Target (outcome)", bool(target_col), target_col or "-", "Required")

    numeric_col = pick_numeric_column(
        df,
        preferred=["yield_tons", "yield", "value", "amount", "score", "prediction", "predicted_yield_tons"],
        exclude=[target_col] if target_col else [],
        min_valid_ratio=0.15,
    )
    numeric_ratio = (numeric_valid_ratio(df, numeric_col) * 100.0) if numeric_col else 0.0
    numeric_mapped = f"{numeric_col} ({numeric_ratio:.0f}% numeric)" if numeric_col else "-"
    add_row("Numeric measure", bool(numeric_col), numeric_mapped, "Required")

    category_col = pick_category_column(
        df,
        exclude=[target_col, numeric_col] if target_col and numeric_col else ([target_col] if target_col else ([numeric_col] if numeric_col else [])),
        preferred_tokens=["crop", "type", "category", "indicator", "state", "region", "country", "district"],
    )
    if category_col:
        distinct = int(df[category_col].dropna().astype(str).nunique())
        add_row("Category", True, f"{category_col} ({distinct} classes)", "Required")
    else:
        add_row("Category", False, "-", "Required")

    geo_col = geography_column(df)
    add_row("Place (country, region, district)", bool(geo_col), geo_col or "-", "Recommended")

    seed_col = pick_column(df, ["improved_seed_used", "improved_seed", "seed_improved"])
    if seed_col:
        add_row("Seed adoption", True, seed_col, "Optional")

    date_candidates: list[tuple[float, str]] = []
    for col in df.columns:
        norm = normalize_column_name(str(col))
        if not any(token in norm for token in ["date", "time", "year", "month", "week", "period", "season"]):
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format")
            parsed_ratio = float(
                pd.to_datetime(df[col], errors="coerce").notna().mean()
            )
        if parsed_ratio >= 0.2:
            date_candidates.append((parsed_ratio, str(col)))
    date_candidates.sort(key=lambda item: item[0], reverse=True)
    if date_candidates:
        ratio, best_date_col = date_candidates[0]
        add_row("Time", True, f"{best_date_col} ({ratio * 100:.0f}% parseable)", "Recommended")
    else:
        add_row("Time", False, "-", "Recommended")

    total_cells = int(df.shape[0] * max(df.shape[1], 1))
    filled_cells = int(df.notna().sum().sum())
    completeness = (filled_cells / total_cells * 100.0) if total_cells else 0.0
    add_row("Completeness", completeness >= 60.0, f"{completeness:.1f}% non-missing", "Optional")

    return rows


def detect_adoption_metrics(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Detect available adoption/practice metrics in the dataset.
    
    Returns list of (column_name, display_label, metric_type) tuples.
    """
    adoption_patterns = [
        (["improved_seed_used", "improved_seed", "seed_improved"], "Improved seed"),
        (["irrigation_used", "irrigation", "uses_irrigation"], "Irrigation"),
        (["fertilizer_used", "uses_fertilizer", "fertilizer"], "Fertilizer use"),
        (["pesticide_used", "uses_pesticide", "pesticide"], "Pesticide use"),
        (["organic_fertilizer", "uses_organic"], "Organic inputs"),
    ]
    
    detected = []
    for candidates, label in adoption_patterns:
        col = pick_column(df, candidates)
        if col is not None:
            detected.append((col, label, "adoption"))
    
    return detected


def adoption_metric_donut(df: pd.DataFrame, metric_col: str, metric_label: str) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Generic adoption metric visualization (not just seed).
    
    Returns (adoption_rate, valid_df, adopters_df).
    """
    if metric_col not in df.columns:
        return 0.0, df, df.iloc[0:0].copy()
    
    series = df[metric_col].astype(str).str.strip().str.lower()
    yes_values = {"yes", "true", "1", "y"}
    no_values = {"no", "false", "0", "n"}
    valid_mask = series.isin(yes_values | no_values)
    valid_df = df.loc[valid_mask].copy()
    valid_series = series.loc[valid_mask]
    adoption_rate = float(valid_series.isin(yes_values).mean() * 100.0) if len(valid_series) else 0.0
    adopters_df = valid_df.loc[valid_series.isin(yes_values)].copy()

    return adoption_rate, valid_df, adopters_df


def crop_distribution_legend(df: pd.DataFrame) -> list[dict[str, object]]:
    crop_col = pick_column(df, ["crop", "crop_type", "main_crop"])
    if crop_col is None or df.empty:
        return []

    shares = df[crop_col].astype(str).value_counts(normalize=True)
    if shares.empty:
        return []
    palette = CHART_PALETTE

    legend_items: list[dict[str, object]] = []
    for idx, (name, share) in enumerate(shares.items()):
        color = palette[idx % len(palette)]
        legend_items.append({"crop": str(name), "share_pct": float(share) * 100.0, "color": color})

    return legend_items


def adoption_crop_segments(adoption_rate: float, adopters_df: pd.DataFrame) -> list[dict[str, float | str]]:
    crop_col = pick_column(adopters_df, ["crop", "crop_type", "main_crop"])
    if adoption_rate <= 0:
        return []
    if crop_col is None or adopters_df.empty:
        return [{"crop": "Adoption", "color": "#52d273", "start": 0.0, "length": float(adoption_rate)}]

    palette = CHART_PALETTE
    shares = adopters_df[crop_col].astype(str).value_counts(normalize=True)
    segments: list[dict[str, float | str]] = []
    start = 0.0
    for idx, (name, share) in enumerate(shares.items()):
        length = float(adoption_rate) * float(share)
        end = min(100.0, start + length)
        segments.append(
            {
                "crop": str(name),
                "color": palette[idx % len(palette)],
                "start": start,
                "length": max(0.0, end - start),
            }
        )
        start = end
        if start >= adoption_rate:
            break
    return segments


def _as_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def model_readiness_message(metrics: dict) -> tuple[str, str]:
    """Verdict on the model, judged against baselines rather than fixed error cut-offs.

    Error is in the target's own units (kg/ha, t/ha, a price), so an absolute threshold
    cannot mean the same thing across datasets. The forward-looking check against
    carrying the previous season forward is the most honest benchmark, so it wins when
    the metrics include it.
    """
    holdout = metrics.get("temporal_holdout") if isinstance(metrics.get("temporal_holdout"), dict) else {}
    persistence = holdout.get("persistence_metrics") if isinstance(holdout.get("persistence_metrics"), dict) else {}
    holdout_mae = _as_float(holdout.get("mae"))
    persistence_mae = _as_float(persistence.get("mae"))
    if holdout_mae is not None and persistence_mae:
        if holdout_mae < persistence_mae:
            gain = 100 * (1 - holdout_mae / persistence_mae)
            return (
                "Beats naive forecast",
                f"On later seasons it never saw, its average error is {gain:.0f}% lower than "
                "carrying the previous season forward.",
            )
        return (
            "No better than naive",
            "On later seasons it never saw, it does not beat carrying the previous season forward. "
            "Treat predictions as a summary of recent history, not a forecast.",
        )

    mae = _as_float(metrics.get("mae"))
    baseline = metrics.get("baseline_metrics") if isinstance(metrics.get("baseline_metrics"), dict) else {}
    baseline_mae = _as_float(baseline.get("mae"))
    if mae is not None and baseline_mae and mae >= baseline_mae:
        return (
            "Needs more signal",
            "The learned model is not beating a simple median baseline yet. Collect more records before relying on predictions.",
        )

    if float(metrics.get("n_train", 0) or 0) < 100:
        return (
            "Early-stage model",
            "This model is trained on a small sample. Treat predictions as directional guidance, not final decisions.",
        )

    if mae is not None and baseline_mae:
        gain = 100 * (1 - mae / baseline_mae)
        return (
            "Beats median baseline",
            f"Average error is {gain:.0f}% lower than always predicting the median. There is no "
            "time-based check for this dataset, so treat that figure as optimistic.",
        )

    return ("Experimental model", "Model quality is still evolving. Use alongside field judgment.")


def format_year(value: object) -> str:
    number = _as_float(value)
    return str(int(number)) if number is not None else "?"


def render_forward_looking_evaluation(metrics: dict) -> None:
    """Show how the model scores on seasons it never saw, against realistic baselines.

    The headline metrics come from a random train/test split, which is optimistic on a
    country/crop/year panel: 2019 and 2020 rows for the same series land on opposite sides
    of the split. This section reports the split that matches how the model would be used
    - trained on past seasons, scored on later ones - and compares it against carrying the
    previous season's value forward, which is the benchmark a practitioner would apply.
    """
    holdout = metrics.get("temporal_holdout")
    if not isinstance(holdout, dict):
        return

    train_from, train_to = holdout.get("train_periods", [None, None])
    test_from, test_to = holdout.get("test_periods", [None, None])
    with st.container(border=True):
        st.markdown("**Forward-looking check**")
        st.caption(
            f"Trained on {format_year(train_from)}-{format_year(train_to)}, scored on "
            f"{format_year(test_from)}-{format_year(test_to)} "
            f"({format_num(holdout.get('n_test'), 0)} records it never saw), against the naive "
            "forecasts a practitioner would otherwise use."
        )

        rows = [
            {
                "Approach": f"{model_display_label(holdout.get('model_type'))} (this model)",
                "MAE": holdout.get("mae"),
                "RMSE": holdout.get("rmse"),
                "R2": holdout.get("r2"),
            }
        ]
        persistence = holdout.get("persistence_metrics")
        if isinstance(persistence, dict):
            rows.append(
                {
                    "Approach": "Carry previous season forward",
                    "MAE": persistence.get("mae"),
                    "RMSE": persistence.get("rmse"),
                    "R2": persistence.get("r2"),
                }
            )
        baseline = holdout.get("baseline_metrics")
        if isinstance(baseline, dict):
            rows.append(
                {
                    "Approach": "Median of training data",
                    "MAE": baseline.get("mae"),
                    "RMSE": baseline.get("rmse"),
                    "R2": baseline.get("r2"),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.1f"),
                "RMSE": st.column_config.NumberColumn(format="%.1f"),
                "R2": st.column_config.NumberColumn("R²", format="%.3f"),
            },
        )


def model_display_label(model_type: object) -> str:
    labels = {
        "random_forest": "Random forest",
        "extra_trees": "Extra trees",
    }
    return labels.get(str(model_type), pretty_column_label(str(model_type))) if model_type else "Model"


# -----------------------------------------------------------------------------
# Analysis page helpers (Crop Insights, Data Quality)
# -----------------------------------------------------------------------------
# Categorical slots for multi-series charts, validated (lightness band, CVD and
# normal-vision separation, contrast) against this app's own chart surfaces.
SERIES_PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
SERIES_PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SERIES_PALETTE = SERIES_PALETTE_DARK if IS_DARK_THEME else SERIES_PALETTE_LIGHT
AXIS_INK = THEME["muted"]
ANALYSIS_MAX_SERIES = len(SERIES_PALETTE)
SEVERITY_LABELS = {"high": "High", "medium": "Medium", "low": "Low"}


def geography_column(df: pd.DataFrame) -> str | None:
    """Place column for peer comparisons: country first, then sub-national units."""
    return pick_column(
        df,
        ["country", "country_name", "area", "state", "state_region", "region", "district", "district_lga", "lga"],
    ) or pick_geography_column(df)


def category_columns(df: pd.DataFrame, max_unique: int = 100) -> list[str]:
    """Text columns worth grouping by: 2..max_unique values, no personal data."""
    out: list[str] = []
    for col in df.columns:
        name = str(col)
        if is_risky_column(name) or pd.api.types.is_numeric_dtype(df[col]):
            continue
        unique = int(df[col].dropna().astype(str).str.strip().nunique())
        if 2 <= unique <= max_unique:
            out.append(name)
    return out


def time_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        norm = normalize_column_name(str(col))
        if not any(tok in norm for tok in ["year", "date", "season", "time", "period"]):
            continue
        if any(tok in norm for tok in ["yield", "prev", "kg", "ha"]):
            continue
        if period_values(df[col]).notna().mean() >= 0.5:
            out.append(str(col))
    return out


def option_index(options: list[Any], value: Any, fallback: int = 0) -> int:
    return options.index(value) if value in options else fallback


def axis_title(column: str) -> str:
    unit = measure_unit(column)
    return f"{measure_label(column)} ({unit})" if unit else measure_label(column)


@st.cache_data(show_spinner="Checking data quality...", max_entries=8)
def cached_quality_report(df: pd.DataFrame, key_columns: tuple[str, ...], group_column: str | None) -> dict[str, Any]:
    return quality_report(df, key_columns=list(key_columns), group_column=group_column)


@st.cache_data(show_spinner="Summarising...", max_entries=16)
def cached_summary(
    df: pd.DataFrame,
    value_col: str,
    group_col: str | None,
    time_col: str | None,
    geo_col: str | None,
    focus: str | None,
    quality: dict[str, Any] | None,
) -> list[dict[str, str]]:
    return write_summary(df, value_col, group_col, time_col, geo_col, focus, quality)


def default_quality_keys(df: pd.DataFrame) -> list[str]:
    geo = geography_column(df)
    group = pick_column(df, ["crop", "crop_type", "main_crop"])
    times = time_columns(df)
    keys = [c for c in [geo, group, times[0] if times else None] if c]
    return keys if len(keys) == 3 else []


def quality_group_default(df: pd.DataFrame) -> str | None:
    return pick_column(df, ["crop", "crop_type", "main_crop", "commodity", "item"])


# -----------------------------------------------------------------------------
# Shared page chrome
# -----------------------------------------------------------------------------
# Inline SVG icons for the KPI cards. Material Symbols cannot be placed inside the
# custom card markup, so the cards carry their own small icon set.
KPI_ICONS: dict[str, str] = {
    "records": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M3 17h2v4H3zm4-6h2v10H7zm4-4h2v14h-2zm4-8h2v18h-2zm4 12h2v6h-2z"/></svg>',
    "average": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 16l4-4 4 4 8-8"/><path d="M4 20h16"/></svg>',
    "place": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2a7 7 0 0 0-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 0 0-7-7zm0 9.5A2.5 2.5 0 1 1 14.5 9 2.5 2.5 0 0 1 12 11.5z"/></svg>',
    "category": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M5 4h14v3H5zm0 6h14v3H5zm0 6h14v3H5z"/></svg>',
    "complete": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm-1 14.5l-3.5-3.5 1.41-1.41L11 13.67l5.59-5.59L18 9.5 11 16.5z"/></svg>',
    "error": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 3a9 9 0 1 0 9 9 9 9 0 0 0-9-9zm1 13h-2v-2h2zm0-4h-2V7h2z"/></svg>',
    "spike": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M13 2 3 14h7l-1 8 10-12h-7z"/></svg>',
    "people": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M16 11a4 4 0 1 0-4-4 4 4 0 0 0 4 4zm-8 0a3 3 0 1 0-3-3 3 3 0 0 0 3 3zm8 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4zm-8 1c-2.33 0-7 1.17-7 3.5V19h7v-1.5A4.68 4.68 0 0 1 10 14.6 8.55 8.55 0 0 0 8 14z"/></svg>',
    "list": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M4 4h16v2H4zm0 5h16v2H4zm0 5h10v2H4zm0 5h10v2H4z"/></svg>',
    "split": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M5 3h14a2 2 0 0 1 2 2v14l-4-3-4 3-4-3-4 3V5a2 2 0 0 1 2-2z"/></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 14a4 4 0 1 1 4-4 4 4 0 0 1-4 4zm0-6a2 2 0 1 0 2 2 2 2 0 0 0-2-2z"/></svg>',
    "model": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M4 4h16v4H4zm0 6h7v10H4zm9 0h7v10h-7z"/></svg>',
    "baseline": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M3 11h18v2H3zm4-6h10v2H7zm0 12h10v2H7z"/></svg>',
    "features": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M4 5h16v2H4zm0 6h16v2H4zm0 6h10v2H4z"/></svg>',
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2 3 7v6c0 5 3.8 9.74 9 11 5.2-1.26 9-6 9-11V7zm0 2.18 7 3.11V13c0 4.07-2.83 7.92-7 9.06C7.83 20.92 5 17.07 5 13V7.29z"/></svg>',
    "signals": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M11 2v20a9 9 0 1 1 0-20zm2 0a9 9 0 0 1 0 20z"/></svg>',
    "check": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M10 17 5 12l1.41-1.41L10 14.17l7.59-7.59L19 8z"/></svg>',
    "alert": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M1 21h22L12 2zm12-3h-2v-2h2zm0-4h-2v-4h2z"/></svg>',
}

BRAND_SERIES = "#52d273"
GAUGE_ARC = "M 20 130 A 104 104 0 0 1 228 130"

district_filter_key = "global_district_filter"
crop_filter_key = "global_crop_filter"
district_filter_choice_key = "global_district_filter_choice"
crop_filter_choice_key = "global_crop_filter_choice"

active_df = get_active_frame()


def lower_first(text: str) -> str:
    """Lower-case the first letter only, so units such as "US$" survive mid-sentence."""
    return text[:1].lower() + text[1:] if text else text


def plural_label(label: str) -> str:
    if label.endswith("y") and not label.endswith(("ay", "ey", "oy")):
        return label[:-1] + "ies"
    return label if label.endswith("s") else label + "s"


def reset_global_filters() -> None:
    for key in (district_filter_key, crop_filter_key, district_filter_choice_key, crop_filter_choice_key):
        st.session_state[key] = "All"


def gauge_html(rate: float, label: str, segments: list[dict[str, float | str]]) -> str:
    """Half-ring gauge. Segments fill in with a CSS animation, so no script is needed."""
    segment_paths = "".join(
        f'<path class="gauge-seg" d="{GAUGE_ARC}" pathLength="100" stroke="{seg["color"]}" '
        f'stroke-dashoffset="-{float(seg["start"]):.4f}" '
        f'style="--len:{float(seg["length"]):.4f};--delay:{int(float(seg["start"]) * 8)}ms"></path>'
        for seg in segments
    )
    safe_label = html.escape(label)
    return (
        f'<figure class="gauge" role="img" aria-label="{safe_label}: {rate:.1f}%">'
        '<svg class="gauge-svg" viewBox="0 0 248 140" aria-hidden="true">'
        f'<path class="gauge-track" d="{GAUGE_ARC}" pathLength="100"></path>{segment_paths}'
        "</svg>"
        f'<figcaption class="gauge-center"><span class="gauge-value">{rate:.1f}%</span>'
        f'<span class="gauge-label">{safe_label}</span></figcaption>'
        "</figure>"
    )


def legend_chips_html(entries: list[dict[str, object]]) -> str:
    chips = "".join(
        f'<span class="legend-chip"><span class="legend-dot" style="--dot:{entry["color"]};"></span>'
        f'{html.escape(str(entry["crop"]))} {float(entry["share_pct"]):.0f}%</span>'
        for entry in entries
    )
    return f'<div class="crop-legend">{chips}</div>'


def ranked_bar_chart(frame: pd.DataFrame, category_title: str, value_title: str, value_format: str) -> alt.Chart:
    """Horizontal bars, largest first. ``frame`` has ``category`` and ``value`` columns."""
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4, color=BRAND_SERIES, height={"band": 0.72})
        .encode(
            y=alt.Y("category:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, labelOverlap=False)),
            x=alt.X("value:Q", title=value_title),
            tooltip=[
                alt.Tooltip("category:N", title=category_title),
                alt.Tooltip("value:Q", title=value_title, format=value_format),
            ],
        )
        .properties(height=max(140, 28 * len(frame)))
        .configure_view(stroke=None)
        .configure_axis(gridOpacity=0.18, labelColor=AXIS_INK, titleColor=AXIS_INK)
    )


def render_sidebar() -> None:
    upload = session_upload()
    if upload is not None:
        with st.sidebar.container(border=True):
            st.markdown(f":material/lock: **Your upload**  \n`{upload['name']}`")
            st.caption("Private to this browser session and never saved.")
            if st.button("Back to demo data", key="sidebar_clear_upload", icon=":material/undo:", width="stretch"):
                clear_session_upload()
                st.rerun()

    with st.sidebar.expander("Pipeline status", icon=":material/monitor_heart:"):
        for label, path in pipeline_artifacts():
            if path.exists():
                st.markdown(f":green[:material/check_circle:] {label}")
            else:
                st.markdown(f":red[:material/cancel:] {label} (missing)")
        st.caption(f"Last run: {last_run_label()}")

    render_admin_unlock()


# -----------------------------------------------------------------------------
# Page: Dashboard
# -----------------------------------------------------------------------------
def page_dashboard() -> None:
    settings = app_settings()
    upload = session_upload()
    has_data = active_df is not None and not active_df.empty

    chips: list[tuple[str, str]] = []
    if has_data:
        chips += [("Records", format_num(len(active_df), 0)), ("Fields", format_num(active_df.shape[1], 0))]
    else:
        chips.append(("Dataset", "Missing"))
    chips.append(("Source", "Your upload") if upload is not None else ("Pipeline run", last_run_label()))
    if has_data:
        detected_target = infer_target_column(active_df)
        if detected_target:
            chips.append(("Target", pretty_column_label(detected_target)))

    render_page_header(
        "Dashboard",
        "Headline figures for the active dataset, the patterns AgroSynapse detects in it, "
        "and how its main measure is distributed.",
        chips=chips,
        eyebrow="At a glance",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)
    if upload is not None:
        st.info(upload_privacy_note(upload), icon=":material/lock:")

    if not has_data:
        st.info(
            "No dataset is available yet. Upload a CSV to explore it, or run the pipeline to build the demo data.",
            icon=":material/database:",
        )
        st.page_link(UPLOADS_PAGE, label="Go to uploads", icon=":material/upload_file:")
        return

    # --- Filters ---------------------------------------------------------------
    district_col = geography_column(active_df)
    crop_col = pick_column(active_df, ["crop", "crop_type", "main_crop"])
    secondary_col = crop_col or pick_category_column(
        active_df,
        exclude=[district_col] if district_col else [],
        preferred_tokens=["crop", "type", "category", "indicator", "soil", "state", "region", "country"],
    )
    district_label = pretty_column_label(district_col) if district_col else "Region"
    secondary_label = pretty_column_label(secondary_col) if secondary_col else "Category"

    def filter_options(column: str | None) -> list[str]:
        if not column:
            return ["All"]
        return ["All"] + sorted(active_df[column].dropna().astype(str).unique().tolist())

    districts = filter_options(district_col)
    crops = filter_options(secondary_col)
    for value_key, choice_key, options in (
        (district_filter_key, district_filter_choice_key, districts),
        (crop_filter_key, crop_filter_choice_key, crops),
    ):
        if st.session_state.get(value_key) not in options:
            st.session_state[value_key] = "All"
        if st.session_state.get(choice_key) not in options:
            st.session_state[choice_key] = st.session_state[value_key]

    selected_district = "All"
    selected_crop = "All"
    if district_col or secondary_col:
        with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
            if district_col:
                selected_district = st.selectbox(district_label, districts, key=district_filter_choice_key, width=260)
            if secondary_col:
                selected_crop = st.selectbox(secondary_label, crops, key=crop_filter_choice_key, width=260)
            st.button(
                "Reset filters",
                key="dashboard_reset_filters",
                icon=":material/filter_alt_off:",
                on_click=reset_global_filters,
                disabled=selected_district == "All" and selected_crop == "All",
            )
    st.session_state[district_filter_key] = selected_district
    st.session_state[crop_filter_key] = selected_crop

    filtered_df = active_df
    if district_col and selected_district != "All":
        filtered_df = filtered_df[filtered_df[district_col].astype(str) == selected_district]
    if secondary_col and selected_crop != "All":
        filtered_df = filtered_df[filtered_df[secondary_col].astype(str) == selected_crop]
    if filtered_df.empty:
        st.warning("No records match these filters.", icon=":material/filter_alt_off:")
        return

    # --- Headline figures ------------------------------------------------------
    yield_col = pick_column(
        filtered_df,
        ["yield_kg_per_ha", "yield_t_per_ha", "yield_tons", "yield", "yield_kg", "actual_yield_tons"],
    )
    focus_col = yield_col or pick_numeric_column(
        filtered_df,
        preferred=["value", "target", "amount", "score", "prediction", "predicted_yield_tons"],
    )
    focus_series = (
        pd.to_numeric(filtered_df[focus_col], errors="coerce").dropna() if focus_col else pd.Series(dtype="float64")
    )
    focus_label = pretty_column_label(focus_col) if focus_col else "Primary measure"
    trend_time_col = next(iter(time_columns(filtered_df)), None)
    total_cells = int(filtered_df.shape[0] * max(filtered_df.shape[1], 1))
    completeness = (int(filtered_df.notna().sum().sum()) / total_cells * 100.0) if total_cells else 0.0

    records_note = "In the current filter"
    if trend_time_col:
        periods = period_values(filtered_df[trend_time_col]).dropna()
        if not periods.empty:
            records_note = f"Covering {int(periods.min())}-{int(periods.max())}"

    def distinct_count(column: str | None) -> int:
        return int(filtered_df[column].dropna().astype(str).nunique()) if column else 0

    kpi_cols = st.columns(4, gap="small")
    with kpi_cols[0]:
        kpi_card("Records", format_num(len(filtered_df), 0), records_note, KPI_ICONS["records"], "#52d273")
    with kpi_cols[1]:
        if not focus_series.empty:
            kpi_card(
                f"Average {lower_first(focus_label)}",
                format_num(focus_series.mean()),
                f"Median {format_num(focus_series.median())}",
                KPI_ICONS["average"],
                "#34c2a6",
            )
        else:
            kpi_card("Completeness", f"{completeness:.1f}%", "Non-missing cells", KPI_ICONS["complete"], "#34c2a6")
    with kpi_cols[2]:
        if district_col:
            kpi_card(
                plural_label(district_label),
                format_num(distinct_count(district_col), 0),
                "Distinct in view",
                KPI_ICONS["place"],
                "#49a9f1",
            )
        else:
            kpi_card("Fields", format_num(filtered_df.shape[1], 0), "Columns available", KPI_ICONS["list"], "#49a9f1")
    with kpi_cols[3]:
        if secondary_col:
            kpi_card(
                plural_label(secondary_label),
                format_num(distinct_count(secondary_col), 0),
                "Distinct in view",
                KPI_ICONS["category"],
                "#8479ff",
            )
        else:
            kpi_card("Completeness", f"{completeness:.1f}%", "Non-missing cells", KPI_ICONS["complete"], "#8479ff")

    # --- Insights --------------------------------------------------------------
    kpi_category_col = pick_category_column(
        filtered_df,
        exclude=[focus_col] if focus_col else [],
        preferred_tokens=["crop", "type", "category", "indicator", "state", "region", "country"],
    )
    comparison_candidates: list[str] = []
    if district_col and selected_district == "All":
        comparison_candidates.append(district_col)
    if secondary_col and selected_crop == "All" and secondary_col not in comparison_candidates:
        comparison_candidates.append(secondary_col)
    if kpi_category_col and kpi_category_col not in comparison_candidates:
        comparison_candidates.append(kpi_category_col)

    st.subheader("Insights")
    render_insight_cards(
        build_dashboard_insights(
            filtered_df,
            focus_col,
            comparison_candidates=comparison_candidates,
            date_col=pick_datetime_column(
                filtered_df,
                preferred=["date", "survey_date", "collection_date", "registration_date", "sample_date", "created_at", "timestamp"],
            ),
            category_col=kpi_category_col,
        ),
        per_row=2,
    )

    # --- Trends ----------------------------------------------------------------
    if focus_col:
        st.subheader("Trends")
        if secondary_col and selected_crop == "All":
            st.caption(
                f"These averages pool every {lower_first(secondary_label)}. Pick one above for a like-for-like comparison."
            )
        rank_by = (
            district_col
            if district_col and selected_district == "All"
            else secondary_col if secondary_col and selected_crop == "All" else None
        )
        trend_panel, rank_panel = st.columns(2, gap="medium")
        with trend_panel, st.container(border=True, height="stretch"):
            st.markdown(f"**{focus_label} over time**")
            means = period_means(filtered_df, focus_col, trend_time_col) if trend_time_col else pd.DataFrame()
            if len(means) >= 2:
                trend_chart = (
                    alt.Chart(means)
                    .mark_line(
                        strokeWidth=2.5,
                        interpolate="monotone",
                        color=BRAND_SERIES,
                        point=alt.OverlayMarkDef(size=26, color=BRAND_SERIES),
                    )
                    .encode(
                        x=alt.X("period:Q", title=None, axis=alt.Axis(format="d")),
                        y=alt.Y("value:Q", title=f"Mean {lower_first(focus_label)}"),
                        tooltip=[
                            alt.Tooltip("period:Q", title=pretty_column_label(trend_time_col), format="d"),
                            alt.Tooltip("value:Q", title=f"Mean {lower_first(focus_label)}", format=",.1f"),
                            alt.Tooltip("n:Q", title="Records", format=","),
                        ],
                    )
                    .properties(height=300)
                    .configure_view(stroke=None)
                    .configure_axis(gridOpacity=0.18, labelColor=AXIS_INK, titleColor=AXIS_INK)
                )
                st.altair_chart(trend_chart, width="stretch")
            else:
                st.caption("No usable time column in this view, so there is no trend to draw.")

        with rank_panel, st.container(border=True, height="stretch"):
            if rank_by:
                ranked = (
                    pd.DataFrame(
                        {
                            "category": filtered_df[rank_by].astype(str).str.strip(),
                            "value": pd.to_numeric(filtered_df[focus_col], errors="coerce"),
                        }
                    )
                    .dropna()
                    .groupby("category")["value"]
                    .agg(["mean", "count"])
                )
                ranked = ranked[ranked["count"] >= 3]["mean"].nlargest(10).rename("value").reset_index()
                rank_label = pretty_column_label(rank_by)
                st.markdown(f"**Top {len(ranked)} by average {lower_first(focus_label)}**")
                if ranked.empty:
                    st.caption(f"No {lower_first(rank_label)} has at least three records.")
                else:
                    st.altair_chart(
                        ranked_bar_chart(ranked, rank_label, f"Mean {lower_first(focus_label)}", ",.1f"),
                        width="stretch",
                    )
                    st.caption(f"Grouped by {lower_first(rank_label)}; groups with fewer than three records are left out.")
            else:
                st.markdown("**Ranking**")
                st.caption("Clear a filter to rank the remaining groups against each other.")

    # --- Adoption or composition ----------------------------------------------
    adoption_metrics = detect_adoption_metrics(filtered_df)
    if adoption_metrics:
        metric_col, metric_label, _ = adoption_metrics[0]
        adoption_rate, _, adopters_df = adoption_metric_donut(filtered_df, metric_col, metric_label)
        crop_legend = crop_distribution_legend(adopters_df)
        st.subheader(f"{metric_label} adoption")
        gauge_panel, text_panel = st.columns([1, 1.4], gap="large", vertical_alignment="center")
        with gauge_panel, st.container(border=True):
            st.html(gauge_html(adoption_rate, f"{metric_label} adoption", adoption_crop_segments(adoption_rate, adopters_df)))
        with text_panel:
            st.markdown(
                f"**{adoption_rate:.1f}%** of records that answered report {lower_first(metric_label)}."
            )
            if crop_legend:
                st.markdown(legend_chips_html(crop_legend), unsafe_allow_html=True)
                st.caption("The arc fill is the adoption rate; its colours split adopting records by crop.")
            else:
                st.caption("No adopting records in the current filter.")
    else:
        snapshot_col = pick_category_column(
            filtered_df,
            exclude=[focus_col] if focus_col else [],
            preferred_tokens=["crop", "type", "category", "indicator", "state", "region", "country", "district"],
        )
        if snapshot_col:
            shares = filtered_df[snapshot_col].dropna().astype(str).str.strip().value_counts(normalize=True)
            snapshot_label = pretty_column_label(snapshot_col)
            st.subheader(f"Records by {lower_first(snapshot_label)}")
            chart_panel, text_panel = st.columns([1.4, 1], gap="large", vertical_alignment="center")
            with chart_panel, st.container(border=True):
                top_shares = (shares.head(8) * 100.0).rename("value").rename_axis("category").reset_index()
                st.altair_chart(ranked_bar_chart(top_shares, snapshot_label, "Share of records (%)", ".1f"), width="stretch")
            with text_panel:
                lines = [
                    f"- **{html.escape(str(shares.index[0]))}** is the largest group, with {shares.iloc[0] * 100:.1f}% of records.",
                    f"- {len(shares):,} distinct values in total; the chart shows the top {min(8, len(shares))}.",
                ]
                if not focus_series.empty:
                    lines.append(f"- Median {lower_first(focus_label)} in this view: {format_num(focus_series.median())}.")
                st.markdown("\n".join(lines))

    # --- Distribution ----------------------------------------------------------
    st.subheader("Distribution")
    if focus_series.empty:
        st.info("No numeric measure detected in the current dataset.", icon=":material/info:")
    else:
        distribution_panel, count_panel = st.columns([1.4, 1], gap="medium")
        with distribution_panel, st.container(border=True, height="stretch"):
            st.markdown(f"**{focus_label}**")
            series = focus_series
            display_series = series
            upper_cap = float(series.quantile(0.95))
            outlier_count = int((series > upper_cap).sum())
            capped = outlier_count > 0 and len(series) >= 8
            if capped:
                display_series = series.clip(upper=upper_cap)

            if len(display_series) >= 4:
                edges = np.histogram_bin_edges(display_series, bins="fd")
                maxbins = (
                    int(max(5, min(40, len(edges) - 1)))
                    if 5 <= len(edges) <= 16
                    else int(settings.get("dashboard_hist_bins", 10))
                )
            else:
                maxbins = int(settings.get("dashboard_hist_bins", 8))
            n_obs = int(len(display_series))
            v_range = float(display_series.max() - display_series.min())
            bin_spec = alt.Bin(maxbins=maxbins)
            if n_obs <= 25 and v_range > 0:
                target_bins = max(6, min(12, int(np.sqrt(n_obs) * 2)))
                raw_step = max(0.5, v_range / target_bins)
                magnitude = 10 ** np.floor(np.log10(raw_step))
                steps = np.array([1.0, 2.0, 2.5, 5.0, 10.0]) * magnitude
                larger = steps[steps >= raw_step]
                bin_spec = alt.Bin(step=float(larger[0] if len(larger) else steps[-1]))

            x_title = f"{focus_label} (display capped at {format_num(upper_cap)})" if capped else focus_label
            chart_df = pd.DataFrame({"metric_value": display_series.to_numpy()})
            hist = (
                alt.Chart(chart_df)
                .mark_bar(color="#2fbd66", opacity=0.9, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X("metric_value:Q", bin=bin_spec, title=x_title),
                    y=alt.Y("count():Q", title="Records"),
                    tooltip=[alt.Tooltip("count():Q", title="Records", format=",")],
                )
            )
            stats = pd.DataFrame(
                {"stat": ["Mean", "Median"], "x": [float(display_series.mean()), float(display_series.median())]}
            )
            rules = (
                alt.Chart(stats)
                .mark_rule(strokeWidth=2)
                .encode(
                    x="x:Q",
                    color=alt.Color(
                        "stat:N",
                        scale=alt.Scale(domain=["Mean", "Median"], range=["#f7b267", "#7db7ff"]),
                        legend=alt.Legend(title=None, orient="top", labelColor=AXIS_INK),
                    ),
                    strokeDash=alt.StrokeDash(
                        "stat:N",
                        scale=alt.Scale(domain=["Mean", "Median"], range=[[8, 4], [2, 3]]),
                        legend=None,
                    ),
                    tooltip=[alt.Tooltip("stat:N", title="Statistic"), alt.Tooltip("x:Q", title="Value", format=",.1f")],
                )
            )
            st.altair_chart(
                (hist + rules)
                .properties(height=300)
                .configure_view(stroke=None)
                .configure_axis(gridOpacity=0.18, labelColor=AXIS_INK, titleColor=AXIS_INK),
                width="stretch",
            )
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            st.caption(
                f"Mean {format_num(series.mean())} · median {format_num(series.median())} · "
                f"middle half between {format_num(q1)} and {format_num(q3)}."
            )
            if capped:
                noun = "record is" if outlier_count == 1 else "records are"
                st.caption(
                    f"{outlier_count:,} high-value {noun} drawn at the 95th percentile for readability; "
                    "the figures above use the real values."
                )

        with count_panel, st.container(border=True, height="stretch"):
            group_col = district_col if district_col and selected_district == "All" else None
            group_col = group_col or pick_category_column(
                filtered_df,
                exclude=[focus_col],
                preferred_tokens=["district", "region", "state", "country", "crop", "type", "category", "indicator"],
            )
            if group_col:
                group_label = pretty_column_label(group_col)
                limit = int(settings.get("top_regions_limit", 10))
                counts = (
                    filtered_df[group_col].astype(str).str.strip().value_counts().head(limit)
                    .rename("value").rename_axis("category").reset_index()
                )
                st.markdown(f"**Records by {lower_first(group_label)}**")
                st.altair_chart(ranked_bar_chart(counts, group_label, "Records", ","), width="stretch")
                st.caption(f"Top {len(counts)} shown.")
            else:
                st.markdown("**Records by group**")
                st.caption("No suitable grouping field detected for this dataset.")

    # --- Data checks -----------------------------------------------------------
    diff = data_diff_summary()
    # The diff compares the pipeline's raw and cleaned files; it says nothing about an
    # upload being explored, so hide it rather than show unrelated numbers.
    show_diff = bool(settings.get("show_data_diff", True) and diff and session_upload() is None)

    st.subheader("Data checks")
    st.caption("Whether the dataset has the fields the dashboard relies on, and what the cleaning step changed.")
    tab_labels = ["Field coverage"] + (["Raw to clean changes"] if show_diff else [])
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        st.dataframe(pd.DataFrame(data_contract_check(active_df)), width="stretch", hide_index=True)
    if show_diff:
        with tabs[1]:
            d1, d2, d3 = st.columns(3, gap="small")
            with d1:
                kpi_card("Raw columns", str(diff["raw_cols"]), f"{format_num(diff['raw_shape'][0], 0)} rows")
            with d2:
                kpi_card("Clean columns", str(diff["clean_cols"]), f"{format_num(diff['clean_shape'][0], 0)} rows")
            with d3:
                kpi_card("Prediction columns", str(len(diff["added_by_prediction"])), "Added by the model")

            with st.expander("Field-level changes", icon=":material/difference:"):
                st.markdown(
                    f"- Renamed during normalisation: {len(diff['renamed_pairs'])}\n"
                    f"- Removed by cleaning: {len(diff['removed_after_clean'])}\n"
                    f"- Added by prediction: {len(diff['added_by_prediction'])}"
                )
                if diff["renamed_pairs"]:
                    st.markdown("**Renamed fields**")
                    st.dataframe(
                        pd.DataFrame(diff["renamed_pairs"]).head(25),
                        width="stretch",
                        hide_index=True,
                        column_config={"from": "Original name", "to": "Normalised name"},
                    )
                if diff["removed_after_clean"]:
                    st.markdown("**Removed by cleaning**")
                    st.dataframe(
                        pd.DataFrame({"Column": diff["removed_after_clean"]}).head(25),
                        width="stretch",
                        hide_index=True,
                    )
                if diff["added_by_prediction"]:
                    st.markdown("**Prediction output columns**")
                    st.dataframe(
                        pd.DataFrame({"Column": diff["added_by_prediction"]}),
                        width="stretch",
                        hide_index=True,
                    )


# -----------------------------------------------------------------------------
# Page: Model Performance
# -----------------------------------------------------------------------------
def page_model_performance() -> None:
    settings = app_settings()
    metrics = load_metrics(str(settings["metrics_path"]))
    preds = load_csv(str(settings["predictions_path"]))
    model_skipped = bool(metrics and str(metrics.get("status", "")).lower() == "skipped")

    chips: list[tuple[str, str]] = []
    if metrics:
        if metrics.get("target"):
            chips.append(("Target", pretty_column_label(str(metrics.get("target")))))
        if model_skipped:
            chips.append(("Status", "Training skipped"))
        else:
            if metrics.get("model_type"):
                chips.append(("Model", model_display_label(metrics.get("model_type"))))
            chips.append(("Verdict", model_readiness_message(metrics)[0]))
    else:
        chips.append(("Model", "Missing"))
    chips.append(("Pipeline run", last_run_label()))

    render_page_header(
        "Model performance",
        "How far off the yield forecasts are, what they are compared against, and how "
        "predictions line up with what actually happened.",
        chips=chips,
        eyebrow="Forecasting",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    if session_upload() is not None:
        retrain_hint = (
            "Retraining is turned off on this shared demo; run the app locally to train on your data."
            if is_demo_mode()
            else "Run the pipeline from the Uploads page to train on your data."
        )
        st.info(
            "This page describes the model trained on the demo dataset, not your upload. " + retrain_hint,
            icon=":material/info:",
        )

    if not metrics:
        st.info(
            "No model metrics yet. Run `python scripts/run_pipeline.py` to train a model.",
            icon=":material/model_training:",
        )
    elif model_skipped:
        reason = str(metrics.get("reason", "Target column was unavailable for model training."))
        st.warning(f"Model training was skipped in the latest pipeline run. {reason}", icon=":material/warning:")
        detail_parts = [
            f"{name}: {format_num(metrics.get(key), 0)}"
            for key, name in [("n_rows", "Rows"), ("n_columns", "Columns"), ("n_features", "Features transformed")]
            if metrics.get(key) is not None
        ]
        if detail_parts:
            st.caption(" · ".join(detail_parts))
        st.caption("Choose a valid target on the Uploads page to train and generate fresh predictions.")
    else:
        error_unit = measure_unit(metrics.get("target")) or "target units"
        kpi_specs = [
            ("mae", f"Average error ({error_unit})", "Mean absolute miss; lower is better", "error", "#f7b267", 1),
            ("rmse", f"Typical error ({error_unit})", "Root mean square; large misses weigh more", "spike", "#f39a58", 1),
            ("n_train", "Training records", "Examples the model learned from", "people", "#52d273", 0),
            ("n_test", "Test records", "Held out to check the model", "list", "#49a9f1", 0),
        ]
        kpi_cols = st.columns(4, gap="small")
        for col, (key, label, note, icon, color, digits) in zip(kpi_cols, kpi_specs):
            with col:
                kpi_card(label, format_num(metrics.get(key), digits), note, KPI_ICONS[icon], color)
        if metrics.get("test_size") is not None:
            st.caption(
                f"Headline figures come from a random {float(metrics['test_size']):.0%} hold-out. "
                "The forward-looking check below is the stricter test."
            )

        verdict, verdict_text = model_readiness_message(metrics)
        st.info(f"**{verdict}.** {verdict_text}", icon=":material/insights:")

        render_forward_looking_evaluation(metrics)

        baseline_metrics = metrics.get("baseline_metrics") if isinstance(metrics.get("baseline_metrics"), dict) else {}
        dataset_profile = metrics.get("dataset_profile") if isinstance(metrics.get("dataset_profile"), dict) else {}
        st.subheader("Model details")
        diag_cols = st.columns(3, gap="small")
        with diag_cols[0]:
            kpi_card(
                "Selected model",
                model_display_label(metrics.get("model_type")),
                "Best of the candidates trained",
                KPI_ICONS["model"],
                "#52d273",
            )
        with diag_cols[1]:
            kpi_card(
                f"Median baseline error ({error_unit})",
                format_num(baseline_metrics.get("mae"), 1),
                "Always predicting the median",
                KPI_ICONS["baseline"],
                "#f7b267",
            )
        with diag_cols[2]:
            kpi_card(
                "Encoded features",
                format_num(dataset_profile.get("estimated_encoded_features"), 0),
                "Inputs after preprocessing",
                KPI_ICONS["features"],
                "#49a9f1",
            )
        for warning in dataset_profile.get("warnings", []):
            st.caption(f":material/warning: {warning}")

        candidate_metrics = metrics.get("candidate_metrics")
        if isinstance(candidate_metrics, dict) and candidate_metrics:
            with st.expander("Candidate comparison", icon=":material/leaderboard:"):
                comparison_rows = [
                    {
                        "Model": model_display_label(name),
                        "MAE": values.get("mae"),
                        "RMSE": values.get("rmse"),
                        "R²": values.get("r2"),
                    }
                    for name, values in candidate_metrics.items()
                    if isinstance(values, dict)
                ]
                if baseline_metrics:
                    comparison_rows.append(
                        {
                            "Model": "Median baseline",
                            "MAE": baseline_metrics.get("mae"),
                            "RMSE": baseline_metrics.get("rmse"),
                            "R²": baseline_metrics.get("r2"),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(comparison_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "MAE": st.column_config.NumberColumn(format="%.1f"),
                        "RMSE": st.column_config.NumberColumn(format="%.1f"),
                        "R²": st.column_config.NumberColumn(format="%.3f"),
                    },
                )

    with st.expander("Model configuration", icon=":material/settings:"):
        model_config_name = model_display_label(metrics.get("model_type")) if metrics and not model_skipped else "Random forest"
        encoder_config = metrics.get("encoder") if isinstance(metrics, dict) and isinstance(metrics.get("encoder"), dict) else {}
        st.code(
            f"""
{model_config_name}

Preprocessing:
- Categorical: OneHotEncoder(handle_unknown="ignore", sparse_output=True)
- Rare category grouping: min_frequency={encoder_config.get("min_frequency")}
- Category cap: max_categories={encoder_config.get("max_categories")}
- Numeric: median imputation after numeric coercion
            """.strip(),
            language="text",
        )

    st.subheader("Predictions")
    if model_skipped:
        st.caption("The latest run generated no new predictions because training was skipped.")
    if preds is None or preds.empty:
        st.info("No predictions available yet. Run `python scripts/predict.py`.", icon=":material/info:")
        return

    # predict.py names its output "predicted_<target>", and the target changes with the
    # dataset, so neither column can be found from a fixed list of names. Resolve the
    # actual column from the trained target and pair it with the matching prediction.
    trained_target = str(metrics.get("target")) if isinstance(metrics, dict) else ""
    actual_col = pick_column(preds, [trained_target, "actual", "actual_yield", "yield_tons"])
    pred_col = pick_column(preds, [f"predicted_{trained_target}", "prediction", "predicted", "predicted_yield"])
    if pred_col is None:
        pred_col = next((c for c in preds.columns if str(c).lower().startswith("predicted")), None)

    if actual_col and pred_col:
        pairs = pd.DataFrame(
            {
                "actual": pd.to_numeric(preds[actual_col], errors="coerce"),
                "predicted": pd.to_numeric(preds[pred_col], errors="coerce"),
            }
        ).dropna()
        for label_col in ("country", "crop", "year"):
            if label_col in preds.columns:
                pairs[label_col] = preds.loc[pairs.index, label_col]
        # Thousands of overlapping points add nothing but render time.
        sample = pairs.sample(min(len(pairs), 4000), random_state=7) if len(pairs) > 4000 else pairs
        unit = measure_unit(actual_col)
        unit_suffix = f" ({unit})" if unit else ""
        lo = float(min(sample["actual"].min(), sample["predicted"].min()))
        hi = float(max(sample["actual"].max(), sample["predicted"].max()))

        with st.container(border=True):
            st.markdown("**Predicted against actual**")
            tooltip = [
                alt.Tooltip(c, title=c.capitalize())
                for c in ("country", "crop", "year")
                if c in sample.columns
            ] + [
                alt.Tooltip("actual:Q", title=f"Actual{unit_suffix}", format=",.0f"),
                alt.Tooltip("predicted:Q", title=f"Predicted{unit_suffix}", format=",.0f"),
            ]
            points = (
                alt.Chart(sample)
                .mark_circle(size=22, opacity=0.45, color=SERIES_PALETTE[0])
                .encode(
                    x=alt.X("actual:Q", title=f"Actual{unit_suffix}", scale=alt.Scale(domain=[lo, hi])),
                    y=alt.Y("predicted:Q", title=f"Predicted{unit_suffix}", scale=alt.Scale(domain=[lo, hi])),
                    tooltip=tooltip,
                )
            )
            diagonal = (
                alt.Chart(pd.DataFrame({"actual": [lo, hi], "predicted": [lo, hi]}))
                .mark_line(color=AXIS_INK, strokeDash=[6, 4], strokeWidth=1.5)
                .encode(x="actual:Q", y="predicted:Q")
            )
            st.altair_chart(
                (points + diagonal)
                .properties(height=420)
                .configure_view(stroke=None)
                .configure_axis(gridOpacity=0.18, labelColor=AXIS_INK, titleColor=AXIS_INK),
                width="stretch",
            )
            sampled_note = f" A random {len(sample):,} of {len(pairs):,} are drawn." if len(sample) < len(pairs) else ""
            st.caption(
                f"Points on the dashed line were predicted exactly; distance from it is the error.{sampled_note}"
            )

    with st.expander("Prediction sample", icon=":material/table_rows:"):
        st.dataframe(preds.head(15), width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# Page: Data Explorer
# -----------------------------------------------------------------------------
def render_dataset_view(df: pd.DataFrame, file_name: str, key: str, noun: str = "columns") -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"{df.shape[0]:,} rows · {df.shape[1]:,} {noun}")
        st.space("stretch")
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=file_name,
            mime="text/csv",
            key=key,
            icon=":material/download:",
        )
    st.dataframe(df, width="stretch", height=440)


def page_data_explorer() -> None:
    settings = app_settings()
    raw_exists = Path(str(settings["raw_data_path"])).exists() or Path(str(settings["raw_fallback_path"])).exists()
    render_page_header(
        "Data explorer",
        "Browse and download each stage of the data: the raw source, the cleaned table, and the "
        "engineered features the model trains on.",
        chips=[
            ("Raw", "Available" if raw_exists else "Missing"),
            ("Clean", "Available" if Path(str(settings["clean_data_path"])).exists() else "Missing"),
            ("Features", "Available" if Path(str(settings["features_data_path"])).exists() else "Missing"),
        ],
        eyebrow="Data inspection",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    explorer_upload = session_upload()
    tab_labels = ["Raw data", "Cleaned data", "Features"]
    if explorer_upload is not None:
        st.info(
            upload_privacy_note(explorer_upload)
            + " The other tabs show the demo pipeline's files, which your upload does not change.",
            icon=":material/lock:",
        )
        tabs = st.tabs(["Your upload"] + [f"Demo {lower_first(label)}" for label in tab_labels])
        with tabs[0]:
            upload_raw = explorer_upload["raw"]
            render_dataset_view(upload_raw, explorer_upload["name"], "dl_upload")
            with st.expander("Column profile", icon=":material/view_column:"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Column": upload_raw.columns.astype(str),
                            "Type": upload_raw.dtypes.astype(str).to_numpy(),
                            "Non-empty": upload_raw.notna().sum().to_numpy(),
                            "Missing %": (upload_raw.isna().mean() * 100).round(1).to_numpy(),
                            "Distinct": upload_raw.nunique(dropna=True).to_numpy(),
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Missing %": st.column_config.ProgressColumn(
                            "Missing", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                )
        raw_tab, clean_tab, features_tab = tabs[1:]
    else:
        raw_tab, clean_tab, features_tab = st.tabs(tab_labels)

    with raw_tab:
        raw = load_csv(str(settings["raw_data_path"]))
        if raw is None:
            raw = load_csv(str(settings["raw_fallback_path"]))
        if raw is None:
            st.info(
                f"No raw data found at `{settings['raw_data_path']}` or `{settings['raw_fallback_path']}`. "
                "Upload a CSV on the Uploads page to get started.",
                icon=":material/database:",
            )
        else:
            render_dataset_view(raw, "raw_data.csv", "dl_raw")

    with clean_tab:
        clean = load_csv(str(settings["clean_data_path"]))
        if clean is None:
            st.info(
                f"No cleaned data found at `{settings['clean_data_path']}`. Run the pipeline to generate it.",
                icon=":material/database:",
            )
        else:
            render_dataset_view(clean, "clean_data.csv", "dl_clean")

    with features_tab:
        features = load_csv(str(settings["features_data_path"]))
        if features is None:
            st.info(
                f"No feature set found at `{settings['features_data_path']}`. Run the pipeline to generate it.",
                icon=":material/database:",
            )
        else:
            render_dataset_view(features, "features.csv", "dl_features", noun="features")


# -----------------------------------------------------------------------------
# Page: Uploads
# -----------------------------------------------------------------------------
def step_heading(step: str, title: str, note: str, compact: bool = False) -> None:
    extra = " compact" if compact else ""
    st.markdown(
        f'<div class="upload-panel-heading{extra}">'
        f'<div class="upload-panel-step">{html.escape(step)}</div>'
        f'<div><div class="upload-panel-title">{html.escape(title)}</div>'
        f'<div class="upload-panel-note">{html.escape(note)}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def page_uploads() -> None:
    settings = app_settings()
    upload_header_chips = [("Pipeline run", last_run_label())]
    if not is_demo_mode():
        upload_header_chips.append(
            ("Input file", Path(str(settings["raw_fallback_path"])).name or str(settings["raw_fallback_path"]))
        )
    render_page_header(
        "Uploads",
        "Bring in your own CSV and explore it across every page. The file stays in your browser "
        "session and is never saved unless you choose to.",
        chips=upload_header_chips,
        eyebrow="Data onboarding",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    if is_demo_mode():
        st.info(DEMO_NOTICE, icon=":material/visibility:")

    pipeline_feedback = st.session_state.pop("pipeline_run_feedback", None)
    if pipeline_feedback:
        st.success(pipeline_feedback, icon=":material/check_circle:")
    pipeline_output = st.session_state.get("pipeline_run_output")
    if pipeline_output:
        with st.expander("Last pipeline log", icon=":material/terminal:"):
            st.code(pipeline_output, language="text")

    preview_df: pd.DataFrame | None = None
    inferred_target: str | None = None
    auto_detect_target = True
    with st.container(border=True):
        step_heading("Step 1", "Choose a file", "Upload a CSV, or point the pipeline at a file already on the server.")
        st.markdown('<div class="upload-subsection-label">File source</div>', unsafe_allow_html=True)
        source_file_col, source_path_col = st.columns([1.15, 0.85], gap="large", vertical_alignment="top")
        with source_file_col:
            uploaded_csv = st.file_uploader(
                "CSV file",
                type=["csv"],
                accept_multiple_files=False,
                key=upload_widget_key(),
                help=(
                    "The file is read into your browser session only. Every page switches "
                    "to it until you choose 'Back to demo data'."
                ),
            )
        with source_path_col:
            pipeline_input_path = st.text_input(
                "Pipeline input path",
                value=str(settings["raw_fallback_path"]),
                disabled=is_demo_mode(),
                help="Passed as `--input` to `scripts/run_pipeline.py`.",
            )

        # Removing the file from the widget does not drop the session copy: the widget is
        # also empty whenever this page is re-rendered after visiting another page, and
        # that must not silently switch the visitor back to the demo data.
        if uploaded_csv is not None:
            upload_error = ingest_session_upload(uploaded_csv)
            if upload_error:
                st.error(upload_error)

        active_upload = session_upload()
        if active_upload is not None:
            note_col, action_col = st.columns([3, 1], gap="medium", vertical_alignment="center")
            with note_col:
                st.success(upload_privacy_note(active_upload), icon=":material/lock:")
            with action_col:
                if st.button(
                    "Open dashboard",
                    key="upload_open_dashboard",
                    icon=":material/space_dashboard:",
                    type="primary",
                    width="stretch",
                ):
                    st.switch_page(DASHBOARD_PAGE)
                if st.button("Back to demo data", key="upload_clear", icon=":material/undo:", width="stretch"):
                    clear_session_upload()
                    st.rerun()
            preview_df = active_upload["frame"]
        else:
            preview_df = load_csv(pipeline_input_path.strip())
        inferred_target = infer_target_column(preview_df)

        st.markdown('<div class="upload-subsection-label">Training target</div>', unsafe_allow_html=True)
        target_col, save_setting_col = st.columns([1.22, 0.78], gap="large", vertical_alignment="top")
        with target_col:
            auto_detect_target = st.toggle(
                "Auto-detect target column",
                value=True,
                help="When enabled, the app picks the most likely numeric outcome column for training.",
            )
            if auto_detect_target:
                pipeline_target_column = inferred_target or ""
                if inferred_target:
                    st.caption(f"Resolved target: `{inferred_target}`")
                else:
                    st.caption("No reliable target detected. The pipeline can still run in data-prep mode.")
            else:
                if preview_df is not None and not preview_df.empty:
                    preferred_numeric = candidate_numeric_columns(preview_df, min_valid_ratio=0.2)
                    ordered_columns: list[str] = []
                    for col_name in preferred_numeric + [str(col) for col in preview_df.columns]:
                        if col_name not in ordered_columns:
                            ordered_columns.append(col_name)
                    default_target = inferred_target if inferred_target in ordered_columns else ordered_columns[0]
                    pipeline_target_column = st.selectbox(
                        "Target column",
                        options=ordered_columns,
                        index=ordered_columns.index(default_target),
                        help="Choose the exact target column to train on.",
                    )
                else:
                    pipeline_target_column = st.text_input(
                        "Target column",
                        value=inferred_target or "yield_tons",
                        help="Set the regression target for training (for example: `yield_tons`, `price`, or `value`).",
                    )
        with save_setting_col:
            update_fallback_setting = st.toggle(
                "Remember this input path",
                value=True,
                help="When enabled, Settings will remember this input path as the fallback raw data path.",
            )
            st.caption("Saves the path as the fallback raw data path in Settings.")

        st.markdown('<div class="upload-panel-divider"></div>', unsafe_allow_html=True)
        step_heading(
            "Step 2",
            "Run the pipeline",
            "Save the file for review, or clean, featurise and retrain in one go.",
            compact=True,
        )
        upload_col, run_col = st.columns(2, gap="medium")
        with upload_col:
            if st.button(
                "Save uploaded CSV",
                icon=":material/save:",
                width="stretch",
                disabled=is_demo_mode(),
                help="Disabled on the shared demo." if is_demo_mode() else None,
            ) and guard_write_action("Saving an upload"):
                saved_ok, saved_msg = save_uploaded_csv(uploaded_csv, pipeline_input_path.strip())
                if saved_ok:
                    if update_fallback_setting:
                        updated_settings = app_settings().copy()
                        updated_settings["raw_fallback_path"] = pipeline_input_path.strip()
                        st.session_state["app_settings"] = updated_settings
                        save_app_settings(updated_settings)
                    st.success(saved_msg, icon=":material/check_circle:")
                else:
                    st.error(saved_msg, icon=":material/error:")

        with run_col:
            if st.button(
                "Run pipeline",
                icon=":material/play_arrow:",
                type="primary",
                width="stretch",
                disabled=is_demo_mode(),
                help=(
                    "Retraining replaces the model every visitor sees, so it is disabled on "
                    "the shared demo. Run the app locally to train on your own data."
                )
                if is_demo_mode()
                else None,
            ) and guard_write_action("Running the pipeline"):
                run_input_path = pipeline_input_path.strip()

                if uploaded_csv is not None:
                    saved_ok, saved_msg = save_uploaded_csv(uploaded_csv, run_input_path)
                    if not saved_ok:
                        st.error(saved_msg)
                        st.stop()
                    if update_fallback_setting:
                        updated_settings = app_settings().copy()
                        updated_settings["raw_fallback_path"] = run_input_path
                        st.session_state["app_settings"] = updated_settings
                        save_app_settings(updated_settings)

                with st.spinner("Running the pipeline. This can take a minute..."):
                    run_ok, run_output = run_pipeline_for_input(
                        run_input_path,
                        pipeline_target_column,
                        app_settings(),
                    )

                if run_ok:
                    # The pipeline output now reflects this file; drop the session copy so
                    # the dashboard shows the processed data rather than the raw upload.
                    clear_session_upload()
                    st.session_state["pipeline_run_feedback"] = "Pipeline completed. Dashboard and explorer data have been refreshed."
                    st.session_state["pipeline_run_output"] = run_output
                    load_csv.clear()
                    load_metrics.clear()
                    st.rerun()
                else:
                    st.error("The pipeline run failed. The log is below.", icon=":material/error:")
                    if run_output:
                        st.code(run_output, language="text")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        step_heading("Check", "Readiness", "A quick health check of the selected file.")
        status, status_msg = target_health_message(
            preview_df,
            pipeline_target_column,
            inferred_target=inferred_target,
            auto_detect=auto_detect_target,
        )
        if status == "good":
            st.success(status_msg, icon=":material/check_circle:")
        elif status == "warn":
            st.warning(status_msg, icon=":material/warning:")
        else:
            st.info(status_msg, icon=":material/info:")

        with st.container(horizontal=True):
            st.metric("Rows", format_num(preview_df.shape[0], 0) if preview_df is not None else "-", border=True)
            st.metric("Columns", format_num(preview_df.shape[1], 0) if preview_df is not None else "-", border=True)
            st.metric(
                "Suggested target",
                pretty_column_label(inferred_target) if inferred_target else "-",
                border=True,
            )

        if preview_df is None:
            st.caption("No dataset found yet. Upload a CSV above.")

    if preview_df is not None and not preview_df.empty:
        st.markdown('<div class="stack-gap"></div>', unsafe_allow_html=True)
        preview_overview = dataset_schema_overview(preview_df, target_candidate=inferred_target)
        readiness_score, readiness_band = dataset_readiness_score(preview_df, target_candidate=inferred_target)
        preview_insights = build_upload_preview_insights(preview_df, target_candidate=inferred_target)

        st.subheader("Dataset preview")
        st.caption("Readiness, the likely training target, and the signals AgroSynapse found in this file.")

        preview_stat_cols = st.columns(3, gap="medium")
        with preview_stat_cols[0]:
            readiness_color = "#52d273" if readiness_score >= 75 else "#f7b267" if readiness_score >= 55 else "#f39a58"
            kpi_card(
                "Readiness score",
                f"{readiness_score}/100",
                readiness_band,
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2 3 7v6c0 5 3.8 9.74 9 11 5.2-1.26 9-6 9-11V7zm0 2.18 7 3.11V13c0 4.07-2.83 7.92-7 9.06C7.83 20.92 5 17.07 5 13V7.29z"/></svg>',
                readiness_color,
            )
        with preview_stat_cols[1]:
            target_display = pretty_column_label(preview_overview["target"]) if preview_overview["target"] else "Not detected"
            numeric_caption = f"{preview_overview['numeric_count']} numeric field(s)"
            kpi_card(
                "Best target candidate",
                target_display,
                numeric_caption,
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 14a4 4 0 1 1 4-4 4 4 0 0 1-4 4zm0-6a2 2 0 1 0 2 2 2 2 0 0 0-2-2z"/></svg>',
                "#49a9f1",
            )
        with preview_stat_cols[2]:
            kpi_card(
                "Detected signals",
                str(preview_overview["signals_count"]),
                preview_overview["signals_label"],
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M11 2v20a9 9 0 1 1 0-20zm2 0a9 9 0 0 1 0 20z"/></svg>',
                "#8479ff",
            )

        render_insight_cards(preview_insights, per_row=2)

        with st.expander("First 15 rows", icon=":material/table_rows:"):
            st.dataframe(preview_df.head(15), width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# Page: Crop Insights
# -----------------------------------------------------------------------------
def page_crop_insights() -> None:
    insights_upload = session_upload()
    insights_chips: list[tuple[str, str]] = [("Source", "Your upload" if insights_upload else "Demo panel")]
    if active_df is not None and not active_df.empty:
        insights_chips.append(("Records", format_num(len(active_df), 0)))
    render_page_header(
        "Crop insights",
        "Compare crops and places, follow trends over time, and see where one country or region "
        "stands against its peers. Every finding is computed from the data on screen.",
        chips=insights_chips,
        eyebrow="Descriptive analysis",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)
    if insights_upload is not None:
        st.info(upload_privacy_note(insights_upload), icon=":material/lock:")

    measure_options = (
        candidate_numeric_columns(active_df, min_valid_ratio=0.3) if active_df is not None and not active_df.empty else []
    )
    measure_options = [c for c in measure_options if not is_risky_column(c)]
    if not measure_options:
        st.info(
            "Load a dataset with at least one numeric column to analyse it. Use the Uploads page to add a CSV.",
            icon=":material/database:",
        )
    else:
        df_all = active_df
        group_options = category_columns(df_all)
        time_options = time_columns(df_all)
        geo_default = geography_column(df_all)

        default_measure = pick_column(df_all, ["yield_kg_per_ha"]) or infer_target_column(df_all) or measure_options[0]
        default_group = pick_column(df_all, ["crop", "crop_type", "main_crop"])
        if default_group not in group_options:
            default_group = next((c for c in group_options if c != geo_default), None)

        control_cols = st.columns(4, gap="small")
        with control_cols[0]:
            value_col = st.selectbox(
                "Measure",
                measure_options,
                index=option_index(measure_options, default_measure),
                format_func=axis_title,
                key="ci_measure",
            )
        with control_cols[1]:
            group_choices: list[str | None] = [None] + group_options
            group_col = st.selectbox(
                "Compare by",
                group_choices,
                index=option_index(group_choices, default_group),
                format_func=lambda c: "Nothing (whole dataset)" if c is None else pretty_column_label(c),
                key="ci_group",
            )
        with control_cols[2]:
            time_choices: list[str | None] = [None] + time_options
            time_col = st.selectbox(
                "Time",
                time_choices,
                index=option_index(time_choices, time_options[0] if time_options else None),
                format_func=lambda c: "No time column" if c is None else pretty_column_label(c),
                key="ci_time",
            )
        with control_cols[3]:
            geo_choices: list[str | None] = [None] + [c for c in group_options if c != group_col]
            geo_col = st.selectbox(
                "Place",
                geo_choices,
                index=option_index(geo_choices, geo_default),
                format_func=lambda c: "No place column" if c is None else pretty_column_label(c),
                key="ci_geo",
            )

        filter_cols = st.columns([1.2, 1.4, 1.4], gap="medium", vertical_alignment="bottom")
        focus: str | None = None
        with filter_cols[0]:
            if geo_col:
                places = sorted(df_all[geo_col].dropna().astype(str).str.strip().unique().tolist())
                focus = st.selectbox(
                    f"Focus {pretty_column_label(geo_col).lower()}",
                    places,
                    index=option_index(places, "Nigeria"),
                    key="ci_focus",
                )
        scoped = df_all
        with filter_cols[1]:
            has_iita = bool(group_col) and df_all[group_col].dropna().astype(str).map(iita_crop_group).notna().any()
            if has_iita:
                iita_only = st.toggle(
                    "IITA mandate crops only",
                    value=True,
                    help="Cassava, yam, maize, cowpea, soybean, banana and plantain.",
                    key="ci_iita_only",
                )
                if iita_only:
                    scoped = scoped[scoped[group_col].astype(str).map(iita_crop_group).notna()]
        with filter_cols[2]:
            if time_col:
                years = period_values(df_all[time_col])
                if years.notna().any() and years.min() < years.max():
                    lo, hi = int(years.min()), int(years.max())
                    year_range = st.slider("Years", lo, hi, (lo, hi), key=f"ci_years_{time_col}_{lo}_{hi}")
                    in_range = period_values(scoped[time_col]).between(*year_range)
                    scoped = scoped[in_range]

        unit = measure_unit(value_col)
        label = measure_label(value_col)

        if scoped.empty:
            st.warning("No rows match these filters.", icon=":material/filter_alt_off:")
        else:
            quality_for_summary = cached_quality_report(
                df_all, tuple(default_quality_keys(df_all)), quality_group_default(df_all)
            )
            st.subheader("What the data says")
            render_insight_cards(
                cached_summary(scoped, value_col, group_col, time_col, geo_col, focus, quality_for_summary),
                per_row=2,
            )
            st.caption(
                "Rows for the same place repeat across years, so they are not independent. Treat intervals "
                "and p-values here as a first look, not as a formal test."
            )

            tab_names = ["Compare groups", "Trends", "Place vs peers"]
            compare_tab, trend_tab, peer_tab = st.tabs(tab_names)

            with compare_tab:
                if not group_col:
                    st.info("Choose a column under **Compare by** to compare groups.")
                else:
                    compare_scope = scoped
                    if focus:
                        scope_choice = st.segmented_control(
                            "Rows",
                            ["All places", f"{focus} only"],
                            default="All places",
                            key="ci_compare_scope",
                        )
                        if scope_choice == f"{focus} only":
                            compare_scope = scoped[scoped[geo_col].astype(str).str.strip() == focus]
                    table = compare_groups(compare_scope, value_col, group_col, min_n=3)
                    if table.empty:
                        st.info("Not enough rows per group (at least 3) to compare.")
                    else:
                        shown = table.head(30).copy()
                        base = alt.Chart(shown).encode(
                            y=alt.Y("group:N", sort=shown["group"].tolist(), title=None),
                        )
                        tooltip = [
                            alt.Tooltip("group:N", title=pretty_column_label(group_col)),
                            alt.Tooltip("mean:Q", title="Mean", format=",.0f"),
                            alt.Tooltip("ci_low:Q", title="95% CI low", format=",.0f"),
                            alt.Tooltip("ci_high:Q", title="95% CI high", format=",.0f"),
                            alt.Tooltip("median:Q", title="Median", format=",.0f"),
                            alt.Tooltip("n:Q", title="Rows", format=","),
                        ]
                        interval = base.mark_rule(strokeWidth=2, color=SERIES_PALETTE[0]).encode(
                            x=alt.X("ci_low:Q", title=f"Mean {axis_title(value_col).lower()} with 95% CI"),
                            x2="ci_high:Q",
                            tooltip=tooltip,
                        )
                        dots = base.mark_circle(size=90, opacity=1, color=SERIES_PALETTE[0]).encode(
                            x="mean:Q", tooltip=tooltip
                        )
                        chart = (
                            (interval + dots)
                            .properties(height=max(160, 26 * len(shown)))
                            .configure_view(stroke=None)
                            .configure_axisY(labelOverlap=False)
                            .configure_axis(gridOpacity=0.18, labelColor=AXIS_INK, titleColor=AXIS_INK, labelLimit=260)
                        )
                        st.altair_chart(chart, width="stretch")
                        test = group_difference_test(compare_scope, value_col, group_col)
                        if test is not None:
                            verdict = "clearly differ" if test["p_value"] < 0.05 else "do not clearly differ"
                            st.caption(
                                f"Kruskal-Wallis test across {test['n_groups']} groups ({test['n_rows']:,} rows): "
                                f"H = {test['statistic']:,.1f}, p {format_p(test['p_value'])}. The groups {verdict}."
                            )
                        st.dataframe(
                            table,
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "group": st.column_config.TextColumn(pretty_column_label(group_col)),
                                "n": st.column_config.NumberColumn("Rows", format="%d"),
                                "mean": st.column_config.NumberColumn(f"Mean ({unit})" if unit else "Mean", format="%.1f"),
                                "median": st.column_config.NumberColumn("Median", format="%.1f"),
                                "std": st.column_config.NumberColumn("Std dev", format="%.1f"),
                                "ci_low": st.column_config.NumberColumn("95% CI low", format="%.1f"),
                                "ci_high": st.column_config.NumberColumn("95% CI high", format="%.1f"),
                            },
                        )

            with trend_tab:
                if not time_col:
                    st.info("Choose a **Time** column to see trends.")
                else:
                    trend_scope = scoped
                    if focus:
                        trend_place = st.segmented_control(
                            "Rows",
                            [f"{focus} only", "All places"],
                            default=f"{focus} only",
                            key="ci_trend_scope",
                        )
                        if trend_place != "All places":
                            trend_scope = scoped[scoped[geo_col].astype(str).str.strip() == focus]
                    means = period_means(trend_scope, value_col, time_col, group_col)
                    if means.empty:
                        st.info("No values to plot for this selection.")
                    else:
                        available = (
                            means.groupby("group")["n"].sum().sort_values(ascending=False).index.tolist()
                        )
                        picked = st.multiselect(
                            f"{pretty_column_label(group_col)} to plot" if group_col else "Series",
                            available,
                            default=available[: min(ANALYSIS_MAX_SERIES, len(available))],
                            max_selections=ANALYSIS_MAX_SERIES,
                            key=f"ci_trend_groups_{abs(hash(tuple(available)))}",
                        )
                        indexed = st.toggle(
                            "Index to first year = 100",
                            value=len(picked) > 1,
                            help="Puts groups with very different levels on one scale, so their growth can be compared.",
                            key="ci_trend_index",
                        )
                        plot_df = means[means["group"].isin(picked)].copy()
                        if plot_df.empty:
                            st.info("Pick at least one series.")
                        else:
                            if indexed:
                                first = plot_df.sort_values("period").groupby("group")["value"].transform("first")
                                plot_df["plot_value"] = plot_df["value"] / first.replace(0, np.nan) * 100
                                y_title = f"{label} index (first year = 100)"
                            else:
                                plot_df["plot_value"] = plot_df["value"]
                                y_title = f"Mean {axis_title(value_col).lower()}"
                            color = alt.Color(
                                "group:N",
                                scale=alt.Scale(domain=picked, range=SERIES_PALETTE[: len(picked)]),
                                legend=alt.Legend(title=None, orient="top", labelColor=AXIS_INK, labelLimit=240),
                            )
                            hover = alt.selection_point(fields=["period"], nearest=True, on="pointerover", empty=False)
                            base = alt.Chart(plot_df).encode(
                                x=alt.X("period:Q", title=None, axis=alt.Axis(format="d", labelColor=AXIS_INK)),
                                y=alt.Y("plot_value:Q", title=y_title, axis=alt.Axis(labelColor=AXIS_INK, titleColor=AXIS_INK)),
                                color=color,
                            )
                            lines = base.mark_line(strokeWidth=2, interpolate="monotone")
                            points = base.mark_circle(size=64).encode(
                                opacity=alt.when(hover).then(alt.value(1)).otherwise(alt.value(0)),
                                tooltip=[
                                    alt.Tooltip("group:N", title=pretty_column_label(group_col) if group_col else "Series"),
                                    alt.Tooltip("period:Q", title="Year", format="d"),
                                    alt.Tooltip("value:Q", title=f"Mean ({unit})" if unit else "Mean", format=",.0f"),
                                    alt.Tooltip("n:Q", title="Rows"),
                                ],
                            ).add_params(hover)
                            rule = (
                                alt.Chart(plot_df)
                                .mark_rule(color=AXIS_INK, strokeWidth=1, opacity=0.6)
                                .encode(x="period:Q")
                                .transform_filter(hover)
                            )
                            layers = [lines, points, rule]
                            if len(picked) <= 4:
                                last_points = plot_df.sort_values("period").groupby("group").tail(1)
                                layers.append(
                                    alt.Chart(last_points)
                                    .mark_text(align="left", dx=6, fontSize=11, color=THEME["text"])
                                    .encode(x="period:Q", y="plot_value:Q", text="group:N")
                                )
                            chart = (
                                alt.layer(*layers)
                                .properties(height=380)
                                .configure_view(stroke=None)
                                .configure_axis(gridOpacity=0.18)
                            )
                            st.altair_chart(chart, width="stretch")

                            trends = trend_by_group(
                                trend_scope[trend_scope[group_col].astype(str).str.strip().isin(picked)]
                                if group_col
                                else trend_scope,
                                value_col,
                                time_col,
                                group_col,
                            )
                            if not trends.empty:
                                trends["clear"] = np.where(trends["p_value"] < 0.05, "Yes", "No")
                                st.dataframe(
                                    trends[
                                        ["group", "first_period", "last_period", "fitted_start", "fitted_end",
                                         "pct_per_period", "r2", "p_value", "clear"]
                                    ],
                                    hide_index=True,
                                    width="stretch",
                                    column_config={
                                        "group": st.column_config.TextColumn(
                                            pretty_column_label(group_col) if group_col else "Series"
                                        ),
                                        "first_period": st.column_config.NumberColumn("From", format="%d"),
                                        "last_period": st.column_config.NumberColumn("To", format="%d"),
                                        "fitted_start": st.column_config.NumberColumn("Trend start", format="%.0f"),
                                        "fitted_end": st.column_config.NumberColumn("Trend end", format="%.0f"),
                                        "pct_per_period": st.column_config.NumberColumn("Change per year", format="%+.1f%%"),
                                        "r2": st.column_config.NumberColumn("R²", format="%.2f"),
                                        "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
                                        "clear": st.column_config.TextColumn("Clear trend (p < 0.05)"),
                                    },
                                )
                                st.caption(
                                    "Trend = straight line through the yearly means. Change per year is the slope as "
                                    "a share of the series' average level."
                                )

            with peer_tab:
                if not (geo_col and group_col and focus):
                    st.info("Choose both a **Compare by** column and a **Place** column to benchmark one place.")
                else:
                    window = (
                        st.slider("Recent years to average", 1, 10, 5, key="ci_peer_window") if time_col else None
                    )
                    bench = benchmark_against_peers(
                        scoped, value_col, group_col, geo_col, focus, time_col, recent_periods=window or 5
                    )
                    if bench.empty:
                        st.info(
                            f"{focus} has no groups with at least three other places to compare against "
                            "under the current filters."
                        )
                    else:
                        bench = bench.copy()
                        bench["position"] = np.where(bench["gap_pct"] >= 0, "Above peer median", "Below peer median")
                        bars = (
                            alt.Chart(bench)
                            .mark_bar(cornerRadiusEnd=4, height={"band": 0.7})
                            .encode(
                                y=alt.Y("group:N", sort=bench["group"].tolist(), title=None),
                                x=alt.X("gap_pct:Q", title=f"{focus} vs peer median (%)"),
                                color=alt.Color(
                                    "position:N",
                                    scale=alt.Scale(
                                        domain=["Above peer median", "Below peer median"],
                                        range=[SERIES_PALETTE[0], SERIES_PALETTE[1]],
                                    ),
                                    legend=alt.Legend(title=None, orient="top", labelColor=AXIS_INK),
                                ),
                                tooltip=[
                                    alt.Tooltip("group:N", title=pretty_column_label(group_col)),
                                    alt.Tooltip("focus_value:Q", title=focus, format=",.0f"),
                                    alt.Tooltip("peer_median:Q", title="Peer median", format=",.0f"),
                                    alt.Tooltip("gap_pct:Q", title="Gap (%)", format="+.0f"),
                                    alt.Tooltip("rank:Q", title="Rank"),
                                    alt.Tooltip("n_places:Q", title="Places compared"),
                                ],
                            )
                        )
                        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=AXIS_INK).encode(x="x:Q")
                        chart = (
                            (bars + zero)
                            .properties(height=max(140, 30 * len(bench)))
                            .configure_view(stroke=None)
                            .configure_axisY(labelOverlap=False)
                            .configure_axis(gridOpacity=0.18, labelColor=AXIS_INK, titleColor=AXIS_INK, labelLimit=260)
                        )
                        st.altair_chart(chart, width="stretch")
                        st.dataframe(
                            bench[["group", "focus_value", "peer_median", "gap_pct", "rank", "n_places", "window"]],
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "group": st.column_config.TextColumn(pretty_column_label(group_col)),
                                "focus_value": st.column_config.NumberColumn(
                                    f"{focus} ({unit})" if unit else focus, format="%.0f"
                                ),
                                "peer_median": st.column_config.NumberColumn("Peer median", format="%.0f"),
                                "gap_pct": st.column_config.NumberColumn("Gap", format="%+.0f%%"),
                                "rank": st.column_config.NumberColumn("Rank (1 = highest)", format="%d"),
                                "n_places": st.column_config.NumberColumn("Places compared", format="%d"),
                                "window": st.column_config.TextColumn("Years averaged"),
                            },
                        )
                        st.caption(
                            "Each place is reduced to its own mean over the window, so places with more rows do "
                            "not count more. The peer figure is the median of the other places' means."
                        )


# -----------------------------------------------------------------------------
# Page: Data Quality
# -----------------------------------------------------------------------------
def page_data_quality() -> None:
    quality_upload = session_upload()
    render_page_header(
        "Data quality",
        "The checks to run before trusting a dataset: blanks, duplicates, impossible values, "
        "numbers stored as text, inconsistent labels and extreme values.",
        chips=[
            ("Source", "Your upload" if quality_upload else "Demo panel"),
            ("Rows", format_num(len(active_df), 0) if active_df is not None else "0"),
            ("Columns", format_num(active_df.shape[1], 0) if active_df is not None else "0"),
        ],
        eyebrow="Data validation",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)
    if quality_upload is not None:
        st.info(upload_privacy_note(quality_upload), icon=":material/lock:")

    if active_df is None or active_df.empty:
        st.info("Load a dataset to check it. Use the Uploads page to add a CSV.", icon=":material/database:")
    else:
        dq_df = active_df
        dq_columns = [str(c) for c in dq_df.columns]
        dq_groups = category_columns(dq_df)
        setting_cols = st.columns(2, gap="medium")
        with setting_cols[0]:
            key_cols = st.multiselect(
                "Columns that should identify each record",
                dq_columns,
                default=default_quality_keys(dq_df),
                help="Used to find repeated records, e.g. the same farmer and season entered twice.",
                key="dq_keys",
            )
        with setting_cols[1]:
            dq_group_choices: list[str | None] = [None] + dq_groups
            dq_group = st.selectbox(
                "Judge extreme values within each",
                dq_group_choices,
                index=option_index(dq_group_choices, quality_group_default(dq_df)),
                format_func=lambda c: "Whole dataset" if c is None else pretty_column_label(c),
                help="A cassava yield is only extreme compared with other cassava yields.",
                key="dq_group",
            )

        report = cached_quality_report(dq_df, tuple(key_cols), dq_group)
        issues_df = pd.DataFrame(report["issues"])
        counts = {sev: sum(1 for i in report["issues"] if i["severity"] == sev) for sev in SEVERITY_LABELS}

        kpi_cols = st.columns(4, gap="small")
        with kpi_cols[0]:
            kpi_card(
                "Quality score",
                f"{report['score']}/100",
                "Starts at 100; each finding deducts by severity",
                color="#0ca30c" if counts["high"] == 0 else "#d03b3b",
            )
        with kpi_cols[1]:
            kpi_card("High severity", str(counts["high"]), "Fix before analysis", color="#d03b3b")
        with kpi_cols[2]:
            kpi_card("Medium severity", str(counts["medium"]), "Review and document", color="#fab219")
        with kpi_cols[3]:
            kpi_card("Low severity", str(counts["low"]), "Worth knowing", color=AXIS_INK)

        st.subheader("Findings")
        if issues_df.empty:
            st.success("No issues found by these checks.", icon=":material/check_circle:")
        else:
            issues_view = issues_df.assign(severity=issues_df["severity"].map(SEVERITY_LABELS))
            st.dataframe(
                issues_view,
                hide_index=True,
                width="stretch",
                column_config={
                    "severity": st.column_config.TextColumn("Severity", width="small"),
                    "check": st.column_config.TextColumn("Check"),
                    "column": st.column_config.TextColumn("Column"),
                    "rows_affected": st.column_config.NumberColumn("Rows affected", format="%d"),
                    "detail": st.column_config.TextColumn("What it means", width="large"),
                    "examples": st.column_config.TextColumn("Examples", width="medium"),
                },
            )
            st.download_button(
                "Download findings (CSV)",
                issues_view.to_csv(index=False).encode("utf-8"),
                file_name="data_quality_findings.csv",
                mime="text/csv",
                icon=":material/download:",
            )

        profile = report["profile"]
        st.subheader("Column profile")
        missing = profile[profile["missing_pct"] > 0].sort_values("missing_pct", ascending=False).head(15)
        if not missing.empty:
            missing_chart = (
                alt.Chart(missing)
                .mark_bar(cornerRadiusEnd=4, color=SERIES_PALETTE[0], height={"band": 0.7})
                .encode(
                    y=alt.Y("column:N", sort="-x", title=None),
                    x=alt.X("missing_pct:Q", title="Missing values (%)", scale=alt.Scale(domain=[0, 100])),
                    tooltip=[
                        alt.Tooltip("column:N", title="Column"),
                        alt.Tooltip("missing_pct:Q", title="Missing (%)", format=".1f"),
                    ],
                )
                .properties(height=max(120, 26 * len(missing)))
                .configure_view(stroke=None)
                .configure_axisY(labelOverlap=False)
                .configure_axis(gridOpacity=0.18, labelColor=AXIS_INK, titleColor=AXIS_INK, labelLimit=260)
            )
            st.altair_chart(missing_chart, width="stretch")
        st.dataframe(
            profile,
            hide_index=True,
            width="stretch",
            column_config={
                "column": st.column_config.TextColumn("Column"),
                "type": st.column_config.TextColumn("Read as"),
                "missing_pct": st.column_config.ProgressColumn("Missing", format="%.1f%%", min_value=0, max_value=100),
                "unique": st.column_config.NumberColumn("Distinct values", format="%d"),
                "example": st.column_config.TextColumn("Example"),
            },
        )

        with st.expander("What each check looks for", icon=":material/help:"):
            render_markdown_block(
                """
                - **Empty / many missing values**: blank cells. Decide whether a blank means zero, not collected, or not applicable before filling it in.
                - **Duplicate rows / repeated records**: rows counted twice inflate totals and shrink error bars.
                - **Impossible negative values**: yields, areas, prices, ages and similar quantities cannot be below zero.
                - **Percentage above 100** and **implausible year**: values outside the range the column allows.
                - **Numbers stored as text**: a column that is mostly numbers but has entries like `n/a` or `2,4`; it cannot be averaged until fixed.
                - **Inconsistent labels / extra spaces**: `Maize`, `maize ` and `MAIZE` would be counted as three crops.
                - **Extreme values**: more than 5 robust standard deviations (median absolute deviation) from the typical value within each group, on a log scale for skewed quantities. They may be real; check them for unit or entry errors.
                """
            )


# -----------------------------------------------------------------------------
# Page: Overview
# -----------------------------------------------------------------------------
def page_overview() -> None:
    render_page_header(
        "About AgroSynapse",
        "An analyst's workflow for agricultural data: check it, clean it, describe it and, "
        "where the data supports it, forecast from it.",
        eyebrow="Project",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    st.subheader("What it does")
    feature_cols = st.columns(2, gap="medium")
    features = [
        (DATA_QUALITY_PAGE, "Data quality", "Flags blanks, duplicates, impossible values, numbers stored as text, "
         "inconsistent labels and extreme values in any CSV."),
        (CROP_INSIGHTS_PAGE, "Crop insights", "Compares crops and places with confidence intervals, fits trends, benchmarks "
         "one place against its peers, and writes the findings in plain language."),
        (MODEL_PAGE, "Yield forecasting", "Forecasts next season's yield per country and crop, scored on years the model never saw "
         "and against a naive carry-forward forecast."),
        (UPLOADS_PAGE, "Cleaning pipeline", "Normalises column names, drops KoboToolbox metadata, and keeps identifiers and "
         "target-derived columns out of the model."),
    ]
    for index, (target_page, label, description) in enumerate(features):
        with feature_cols[index % 2], st.container(border=True, height="stretch"):
            st.page_link(target_page, label=f"**{label}**")
            st.caption(description)

    flow_col, data_col = st.columns(2, gap="medium")
    with flow_col:
        st.subheader("How it works")
        render_markdown_block(
            """
            1. **Load** a CSV: the demo panel, or your own upload
            2. **Check** its quality and fix what the checks find
            3. **Describe** it with group comparisons, trends and peer benchmarks
            4. **Model** it: train, evaluate against a naive benchmark, predict
            """
        )
    with data_col:
        st.subheader("The demo data")
        render_markdown_block(
            """
            - 25,305 rows, one per **country, crop and year**: 54 African countries, 23 crops, 1990-2024
            - Built from public **FAOSTAT** and **World Bank** data (CC BY 4.0)
            - National figures, not farm records, so results describe countries rather than farms
            - On 2018-2024, the forecast's average error is about a third lower than "same as last season"
            """
        )


# -----------------------------------------------------------------------------
# Page: Settings
# -----------------------------------------------------------------------------
def page_settings() -> None:
    settings = app_settings()
    demo = is_demo_mode()
    settings_score, settings_readiness_label, settings_recommendations = settings_readiness_summary(settings)
    render_page_header(
        "Settings",
        "File paths and dashboard behaviour. Changes are saved for everyone using this server, "
        "so they are locked on the shared demo.",
        chips=[
            ("Readiness", settings_readiness_label),
            ("Animations", "On" if bool(settings["enable_animations"]) else "Off"),
            ("Data diff", "On" if bool(settings["show_data_diff"]) else "Off"),
            ("Write access", "Locked" if demo else "Enabled"),
        ],
        eyebrow="App controls",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    if demo:
        st.info(DEMO_NOTICE, icon=":material/visibility:")

    settings_health_rows = settings_path_health_rows(settings)
    ready_paths = sum(1 for row in settings_health_rows if row["Status"] == "Ready")
    attention_paths = len(settings_health_rows) - ready_paths

    smart_cols = st.columns(3, gap="small")
    with smart_cols[0]:
        kpi_card(
            "Setup readiness",
            f"{settings_score}/100",
            settings_readiness_label,
            KPI_ICONS["shield"],
            "#52d273" if settings_score >= 85 else "#f7b267",
        )
    with smart_cols[1]:
        kpi_card(
            "Files ready",
            f"{ready_paths}/{len(settings_health_rows)}",
            "Present and readable",
            KPI_ICONS["check"],
            "#34c2a6",
        )
    with smart_cols[2]:
        kpi_card(
            "Needs attention",
            format_num(attention_paths, 0),
            "Missing or unreadable",
            KPI_ICONS["alert"],
            "#f39a58" if attention_paths else "#52d273",
        )

    rec_col, table_col = st.columns([0.85, 1.45], gap="medium")
    with rec_col, st.container(border=True, height="stretch"):
        st.markdown("**Recommendations**")
        st.markdown("\n".join(f"- {item}" for item in settings_recommendations))

        raw_primary_exists = Path(str(settings["raw_data_path"])).expanduser().exists()
        raw_fallback_exists = Path(str(settings["raw_fallback_path"])).expanduser().exists()
        if not raw_primary_exists and raw_fallback_exists:
            if st.button(
                "Use fallback CSV as primary", width="stretch", disabled=demo
            ) and guard_write_action("Changing the raw data path"):
                updated_settings = settings.copy()
                updated_settings["raw_data_path"] = str(settings["raw_fallback_path"])
                st.session_state["app_settings"] = updated_settings
                save_app_settings(updated_settings)
                flash("Primary raw path updated.")

    with table_col, st.container(border=True, height="stretch"):
        st.markdown("**File health**")
        st.dataframe(
            pd.DataFrame(settings_health_rows),
            width="stretch",
            hide_index=True,
            column_config={"Details": st.column_config.TextColumn(width="medium")},
        )

    st.subheader("App settings")
    st.caption("Saved to `data/processed/app_settings.json` and applied on every rerun.")

    with st.form("app_settings_form", clear_on_submit=False):
        st.markdown("**Data paths**")
        path_col_1, path_col_2 = st.columns(2, gap="large")
        with path_col_1:
            raw_data_path = st.text_input("Primary raw data", value=str(settings["raw_data_path"]), disabled=demo)
            raw_fallback_path = st.text_input(
                "Fallback raw data", value=str(settings["raw_fallback_path"]), disabled=demo
            )
            clean_data_path = st.text_input("Cleaned data", value=str(settings["clean_data_path"]), disabled=demo)
            features_data_path = st.text_input(
                "Features", value=str(settings["features_data_path"]), disabled=demo
            )
        with path_col_2:
            metrics_path = st.text_input("Model metrics", value=str(settings["metrics_path"]), disabled=demo)
            predictions_path = st.text_input("Predictions", value=str(settings["predictions_path"]), disabled=demo)
            model_path = st.text_input("Model artifact", value=str(settings["model_path"]), disabled=demo)

        st.markdown("**Dashboard behaviour**")
        slider_col_1, slider_col_2 = st.columns(2, gap="large")
        with slider_col_1:
            dashboard_hist_bins = st.slider(
                "Histogram bins",
                min_value=8,
                max_value=40,
                value=int(settings["dashboard_hist_bins"]),
                step=1,
                disabled=demo,
                help="Used when the data does not suggest a bin count of its own.",
            )
        with slider_col_2:
            top_regions_limit = st.slider(
                "Groups in the record-count chart",
                min_value=5,
                max_value=20,
                value=int(settings["top_regions_limit"]),
                step=1,
                disabled=demo,
            )
        enable_animations = st.toggle("Animations", value=bool(settings["enable_animations"]), disabled=demo)
        show_data_diff = st.toggle(
            "Show raw-to-clean changes on the dashboard", value=bool(settings["show_data_diff"]), disabled=demo
        )

        submitted = st.form_submit_button("Save settings", type="primary", icon=":material/save:", disabled=demo)
        if submitted and guard_write_action("Saving settings"):
            updated = {
                "raw_data_path": raw_data_path.strip(),
                "raw_fallback_path": raw_fallback_path.strip(),
                "clean_data_path": clean_data_path.strip(),
                "features_data_path": features_data_path.strip(),
                "metrics_path": metrics_path.strip(),
                "predictions_path": predictions_path.strip(),
                "model_path": model_path.strip(),
                "dashboard_hist_bins": int(dashboard_hist_bins),
                "top_regions_limit": int(top_regions_limit),
                "enable_animations": bool(enable_animations),
                "show_data_diff": bool(show_data_diff),
            }
            st.session_state["app_settings"] = updated
            save_app_settings(updated)
            flash("Settings saved.")

    with st.container(horizontal=True):
        if st.button(
            "Reset to defaults", icon=":material/restart_alt:", disabled=demo
        ) and guard_write_action("Resetting settings"):
            st.session_state["app_settings"] = DEFAULT_SETTINGS.copy()
            save_app_settings(st.session_state["app_settings"])
            flash("Settings reset to defaults.")
        if st.button("Reload from disk", icon=":material/refresh:"):
            st.session_state["app_settings"] = load_app_settings()
            load_csv.clear()
            load_metrics.clear()
            flash("Settings reloaded.")

    st.subheader("Run it locally")
    st.code(
        """
# 1. Install the pinned dependencies (Python 3.11)
pip install -r requirements.txt

# 2. Build the dataset, train the model and write predictions
python scripts/run_pipeline.py --input data/raw/africa_crop_yield.csv --target yield_kg_per_ha

# 3. Start the dashboard with write access enabled
AGROSYNAPSE_MODE=full streamlit run streamlit_app.py
        """.strip(),
        language="bash",
    )


# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
# Native multipage routing: every page gets its own URL (/crop-insights, ...), so a
# view can be linked to directly and the browser's back button works.
DASHBOARD_PAGE = st.Page(page_dashboard, title="Dashboard", icon=":material/space_dashboard:", default=True)
UPLOADS_PAGE = st.Page(page_uploads, title="Uploads", icon=":material/upload_file:", url_path="uploads")
CROP_INSIGHTS_PAGE = st.Page(page_crop_insights, title="Crop insights", icon=":material/grass:", url_path="crop-insights")
DATA_QUALITY_PAGE = st.Page(page_data_quality, title="Data quality", icon=":material/fact_check:", url_path="data-quality")
MODEL_PAGE = st.Page(
    page_model_performance, title="Model performance", icon=":material/model_training:", url_path="model-performance"
)
NAV_SECTIONS = {
    "Analysis": [DASHBOARD_PAGE, CROP_INSIGHTS_PAGE, DATA_QUALITY_PAGE, MODEL_PAGE],
    "Data": [
        st.Page(page_data_explorer, title="Data explorer", icon=":material/table_view:", url_path="data-explorer"),
        UPLOADS_PAGE,
        st.Page(page_settings, title="Settings", icon=":material/tune:", url_path="settings"),
    ],
    "About": [st.Page(page_overview, title="About AgroSynapse", icon=":material/info:", url_path="about")],
}

if WORDMARK_LOGO_PATH.exists() and SIDEBAR_LOGO_PATH.exists():
    st.logo(str(WORDMARK_LOGO_PATH), size="large", icon_image=str(SIDEBAR_LOGO_PATH))
current_page = st.navigation(NAV_SECTIONS)
st.set_page_config(page_title=f"{current_page.title} · AgroSynapse")
render_sidebar()
if flash_message := st.session_state.pop("flash_message", None):
    st.toast(flash_message, icon=":material/check_circle:")
current_page.run()

st.markdown(
    f'<footer class="footer">© {datetime.now().year} AgroSynapse · Demo data from FAOSTAT and the World Bank '
    "(CC BY 4.0)</footer>",
    unsafe_allow_html=True,
)
