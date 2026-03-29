"""AgroSynapse Dashboard - Yield Intelligence Platform."""

from __future__ import annotations

import csv
import json
from datetime import datetime
import re
from pathlib import Path
from typing import Any
import uuid

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pandas.errors import ParserError

st.set_page_config(
    page_title="AgroSynapse | Yield Intelligence",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

SETTINGS_PATH = Path("data/processed/app_settings.json")
DEFAULT_SETTINGS: dict[str, Any] = {
    "raw_data_path": "data/raw/kobo_export_raw.csv",
    "raw_fallback_path": "data/raw/kobo_export.csv",
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

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

    :root {
        --bg-soft: var(--background-color);
        --card: var(--secondary-background-color);
        --ink: var(--text-color);
        --muted: #8a9490;
        --brand: #1f6b3b;
        --brand-strong: #15502b;
        --line: rgba(127, 207, 159, 0.28);
        --accent: #f7b267;
        --space-1: 0.5rem;
        --space-2: 1rem;
        --space-3: 1.5rem;
        --space-4: 2rem;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], p, li, label, input, textarea, select, button {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6, .hero h1, .panel h4, .kpi-value {
        font-family: 'Space Grotesk', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        letter-spacing: 0.01em;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: var(--space-3);
        padding-left: var(--space-3);
        padding-right: var(--space-3);
        max-width: none;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(127, 207, 159, 0.22);
        background: var(--secondary-background-color);
    }

    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 290px;
        max-width: 320px;
    }

    .hero {
        background: radial-gradient(circle at 85% 25%, #7fcf9f 0%, transparent 35%),
                    linear-gradient(110deg, #1f6b3b 0%, #2b8a4b 100%);
        border: 1px solid #d4e5d8;
        border-radius: 16px;
        padding: var(--space-3) var(--space-4);
        margin-bottom: var(--space-3);
        box-shadow: 0 10px 30px rgba(31, 107, 59, 0.12);
    }

    .hero h1 {
        color: white;
        font-size: 2rem;
        margin: 0;
        letter-spacing: 0.01em;
    }

    .hero p {
        color: #e8f5eb;
        margin-top: var(--space-1);
        margin-bottom: 0;
        max-width: 92%;
        line-height: 1.35;
    }

    .kpi-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-left: 6px solid var(--brand);
        border-radius: 12px;
        padding: var(--space-2) var(--space-3);
        min-height: 112px;
        margin-bottom: var(--space-2);
        animation: cardIn 420ms ease;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.85rem;
        margin-bottom: var(--space-1);
    }

    .kpi-value {
        color: var(--ink);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: var(--space-1);
    }

    .kpi-delta {
        color: #5fcf8e;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .panel {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: var(--space-3);
        margin-bottom: var(--space-3);
        animation: panelIn 460ms ease;
    }

    .panel h4 {
        margin-top: 0;
        margin-bottom: var(--space-1);
        color: var(--ink);
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .footer {
        color: #7a8579;
        text-align: center;
        padding: var(--space-3) 0 var(--space-3);
        font-size: 0.85rem;
    }

    .status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-1);
        margin-bottom: var(--space-3);
    }

    .status-pill {
        background: rgba(95, 207, 142, 0.1);
        border: 1px solid rgba(95, 207, 142, 0.28);
        border-radius: 999px;
        padding: 0.34rem 0.74rem;
        font-size: 0.83rem;
        font-weight: 600;
        color: var(--ink);
    }

    .status-pill-run {
        background: rgba(247, 178, 103, 0.12);
        border: 1px solid rgba(247, 178, 103, 0.32);
    }

    .section-gap {
        margin-top: var(--space-2);
    }

    .dashboard-caption {
        margin-bottom: 0.7rem;
        max-width: 68rem;
        font-size: 0.8rem;
        line-height: 1.45;
    }

    .kpi-row-gap {
        height: var(--space-2);
    }

    .view-gap {
        margin-top: 0.45rem;
        margin-bottom: 0.65rem;
    }

    .stack-gap {
        height: var(--space-3);
    }

    .plain-note {
        background: rgba(127, 207, 159, 0.1);
        border: 1px solid rgba(31, 107, 59, 0.22);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        color: var(--ink);
        line-height: 1.35;
        margin-bottom: 0.95rem;
        max-width: 72rem;
        font-size: 0.78rem;
    }

    [data-testid="stHeaderActionElements"] { display: none !important; }
    .stHeading a { display: none !important; }
    a[href^="#"] { display: none !important; }

    .donut-wrap {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: var(--space-3);
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 240px;
        animation: panelIn 520ms ease;
        position: relative;
        overflow: hidden;
    }

    .donut-wrap::before {
        display: none;
    }

    .donut-shell {
        position: relative;
        width: 220px;
        height: 130px;
        z-index: 2;
    }

    .semi-gauge {
        width: 220px;
        height: 130px;
        display: block;
        overflow: visible;
    }

    .semi-track {
        fill: none;
        stroke: rgba(58, 72, 92, 0.45);
        stroke-width: 18;
        stroke-linecap: butt;
    }

    .semi-fill {
        fill: none;
        stroke: #32c26d;
        stroke-width: 18;
        stroke-linecap: round;
        transition: stroke-dasharray 650ms ease;
    }

    .semi-seg {
        fill: none;
        stroke-width: 18;
        stroke-linecap: butt;
        transition: stroke-dasharray 650ms ease, stroke-dashoffset 650ms ease;
    }

    .donut-center {
        position: absolute;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 6;
        font-weight: 700;
        transition: opacity 350ms ease;
    }

    .donut-center .value {
        font-size: 1.35rem;
        color: var(--ink);
        line-height: 1.1;
    }

    .donut-center .label {
        margin-top: 0.15rem;
        color: var(--muted);
        font-size: 0.78rem;
    }

    .crop-legend {
        margin-top: 0.8rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem 0.55rem;
        margin-bottom: 0.45rem;
    }

    .legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.22rem 0.52rem;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: var(--secondary-background-color);
        font-size: 0.68rem;
        color: var(--ink);
        font-weight: 500;
        opacity: 0;
        transform: translateY(6px) scale(0.98);
        animation: legendIn 420ms ease forwards;
    }

    .legend-chip:nth-child(1) { animation-delay: 40ms; }
    .legend-chip:nth-child(2) { animation-delay: 90ms; }
    .legend-chip:nth-child(3) { animation-delay: 140ms; }
    .legend-chip:nth-child(4) { animation-delay: 190ms; }
    .legend-chip:nth-child(5) { animation-delay: 240ms; }
    .legend-chip:nth-child(6) { animation-delay: 290ms; }
    .legend-chip:nth-child(7) { animation-delay: 340ms; }
    .legend-chip:nth-child(8) { animation-delay: 390ms; }

    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--dot);
    }

    .adoption-copy {
        padding-left: 0.9rem;
        max-width: 44rem;
    }

    .adoption-copy h3 {
        font-size: 1.9rem;
        margin-bottom: 0.45rem;
    }

    .adoption-copy p {
        margin-top: 0.15rem;
        margin-bottom: 0.95rem;
        line-height: 1.45;
        font-size: 0.92rem;
    }

    .legend-note {
        color: var(--muted);
        font-size: 10px;
        line-height: 1.3;
        margin-top: 0.2rem;
        margin-bottom: 0;
    }

    @keyframes cardIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes panelIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes legendIn {
        from { opacity: 0; transform: translateY(6px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.get("app_settings", DEFAULT_SETTINGS).get("enable_animations", True):
    st.markdown(
        """
        <style>
        .kpi-card, .panel, .donut-wrap, .legend-chip { animation: none !important; opacity: 1 !important; transform: none !important; }
        .semi-fill, .donut-center { transition: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame | None:
    file_path = Path(path)
    if not file_path.exists():
        return None

    sample_encodings = ["utf-8", "utf-8-sig", "latin1"]
    sample = ""
    detected_encoding = "utf-8"
    for enc in sample_encodings:
        try:
            with file_path.open("r", encoding=enc, newline="") as sample_file:
                sample = sample_file.read(4096)
            detected_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","

    try:
        return pd.read_csv(file_path, encoding=detected_encoding, sep=sep)
    except ParserError:
        fallback_sep = ";" if sep != ";" else ","
        return pd.read_csv(file_path, encoding=detected_encoding, sep=fallback_sep)


def load_metrics(path: str) -> dict | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
            score += 10

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


def style_chart_axes(ax: plt.Axes, dark: bool = True) -> None:
    if dark:
        bg = "#0f1720"
        spine_col = "#2f3f48"
        tick = "#c9d4ce"
        label = "#dce7e1"
        grid = "#2f3f48"
    else:
        bg = "#f4f8f6"
        spine_col = "#b9c8be"
        tick = "#3f4c44"
        label = "#2f3a34"
        grid = "#d6e1da"

    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color(spine_col)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=tick)
    ax.xaxis.label.set_color(label)
    ax.yaxis.label.set_color(label)
    ax.grid(alpha=0.32, color=grid)


def kpi_card(label: str, value: str, delta: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def app_settings() -> dict[str, Any]:
    settings = st.session_state.get("app_settings")
    if not isinstance(settings, dict):
        settings = DEFAULT_SETTINGS.copy()
        st.session_state["app_settings"] = settings
    return settings


def get_active_frame() -> pd.DataFrame | None:
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


def pipeline_status_items() -> list[str]:
    checks = pipeline_artifacts()
    items: list[str] = []
    for label, path in checks:
        items.append(f"{label}: {'OK' if path.exists() else 'Missing'}")
    return items


def last_run_badge() -> str:
    existing_paths = [path for _, path in pipeline_artifacts() if path.exists()]
    if not existing_paths:
        return "Last Run: N/A"
    latest = max(existing_paths, key=lambda p: p.stat().st_mtime)
    latest_dt = datetime.fromtimestamp(latest.stat().st_mtime)
    return f"Last Run: {latest_dt.strftime('%Y-%m-%d %H:%M:%S')}"


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
    checks = [
        ("Yield", ["yield_tons", "yield", "yield_kg", "actual_yield_tons"], "Required"),
        ("Seed Adoption", ["improved_seed_used", "improved_seed", "seed_improved"], "Required"),
        ("Crop", ["crop", "crop_type", "main_crop"], "Required"),
        ("District/Region", ["district", "region", "state", "location"], "Recommended"),
        ("Soil pH", ["soil_ph", "ph", "soil_ph_level"], "Optional"),
    ]
    rows: list[dict[str, str]] = []
    for label, candidates, level in checks:
        mapped = pick_column(df, candidates)
        rows.append(
            {
                "Field": label,
                "Status": "Found" if mapped else "Missing",
                "Mapped Column": mapped or "-",
                "Priority": level,
            }
        )
    return rows


def seed_adoption_donut(df: pd.DataFrame) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    seed_col = pick_column(df, ["improved_seed_used", "improved_seed", "seed_improved"])
    if seed_col is None:
        return 0.0, df, df.iloc[0:0].copy()

    series = df[seed_col].astype(str).str.strip().str.lower()
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
    palette = [
        "#52d273",
        "#34c2a6",
        "#49a9f1",
        "#8479ff",
        "#d26cf3",
        "#f39a58",
        "#f0627a",
        "#b6d657",
    ]

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

    palette = [
        "#52d273",
        "#34c2a6",
        "#49a9f1",
        "#8479ff",
        "#d26cf3",
        "#f39a58",
        "#f0627a",
        "#b6d657",
    ]
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


def model_readiness_message(metrics: dict) -> tuple[str, str]:
    n_train = float(metrics.get("n_train", 0) or 0)
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")

    if n_train < 100:
        return (
            "Early-stage model",
            "This model is trained on a small sample. Treat predictions as directional guidance, not final decisions.",
        )

    if mae is not None and rmse is not None:
        try:
            mae_v = float(mae)
            rmse_v = float(rmse)
            if mae_v <= 2.5 and rmse_v <= 3.5:
                return ("Strong model", "Prediction error is relatively low for this dataset.")
            if mae_v <= 5.0 and rmse_v <= 7.0:
                return ("Improving model", "Useful for planning, but continue collecting data to improve reliability.")
        except (TypeError, ValueError):
            pass

    return ("Experimental model", "Model quality is still evolving. Use alongside field judgment.")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("AgroSynapse")
st.sidebar.caption("Yield Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Model Performance", "Data Explorer", "Settings"],
)

active_df = get_active_frame()
district_filter_key = "global_filter_district"
crop_filter_key = "global_filter_crop"


def reset_global_filters() -> None:
    st.session_state[district_filter_key] = "All"
    st.session_state[crop_filter_key] = "All"

with st.sidebar.expander("Global Filters", expanded=True):
    if active_df is None:
        st.caption("Add data to enable filters.")
        selected_district = "All"
        selected_crop = "All"
    else:
        district_col = pick_geography_column(active_df)
        crop_col = pick_column(active_df, ["crop", "crop_type", "main_crop"])

        if district_col:
            districts = ["All"] + sorted(active_df[district_col].dropna().astype(str).unique().tolist())
            if st.session_state.get(district_filter_key) not in districts:
                st.session_state[district_filter_key] = "All"
            selected_district = st.selectbox("District", districts, key=district_filter_key)
        else:
            selected_district = "All"
            st.caption("District field not found")

        if crop_col:
            crops = ["All"] + sorted(active_df[crop_col].dropna().astype(str).unique().tolist())
            if st.session_state.get(crop_filter_key) not in crops:
                st.session_state[crop_filter_key] = "All"
            selected_crop = st.selectbox("Crop", crops, key=crop_filter_key)
        else:
            selected_crop = "All"
            st.caption("Crop field not found")

    st.button("Reset Filters", width="stretch", on_click=reset_global_filters)

with st.sidebar.expander("Pipeline Status", expanded=False):
    for item in pipeline_status_items():
        st.write(f"- {item}")
    st.caption(last_run_badge())


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>AgroSynapse</h1>
        <p>Operational analytics workspace for farm survey intelligence, yield forecasting, and agronomic decision support.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if active_df is not None:
    filtered_df = active_df.copy()
    district_col = pick_geography_column(filtered_df)
    crop_col = pick_column(filtered_df, ["crop", "crop_type", "main_crop"])

    if district_col and selected_district != "All":
        filtered_df = filtered_df[filtered_df[district_col].astype(str) == selected_district]
    if crop_col and selected_crop != "All":
        filtered_df = filtered_df[filtered_df[crop_col].astype(str) == selected_crop]
else:
    filtered_df = None


# -----------------------------------------------------------------------------
# Page: Dashboard
# -----------------------------------------------------------------------------
if page == "Dashboard":
    settings = app_settings()
    st.subheader("Dashboard Overview")
    st.markdown(
        '<div class="small-muted dashboard-caption">Pipeline summary of farm records, yield patterns, and data quality. Values refresh after each run.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="plain-note">Use KPI cards for scale, Data Diff for transformation checks, and charts for distribution and regional concentration.</div>',
        unsafe_allow_html=True,
    )

    if filtered_df is None or filtered_df.empty:
        st.info("No survey dataset found yet. Add `data/raw/kobo_export.csv` or run the pipeline to unlock charts.")
    else:
        yield_col = pick_column(filtered_df, ["yield_tons", "yield", "yield_kg", "actual_yield_tons"])
        ph_col = pick_column(filtered_df, ["soil_ph", "ph", "soil_ph_level"])

        total_surveys = int(filtered_df.shape[0])
        total_yield = filtered_df[yield_col].sum() if yield_col else None
        avg_yield = filtered_df[yield_col].mean() if yield_col else None
        avg_ph = filtered_df[ph_col].mean() if ph_col else None

        kpi_row_1 = st.columns(2, gap="large")
        kpi_row_2 = st.columns(2, gap="large")
        with kpi_row_1[0]:
            kpi_card("Total Surveys", format_num(total_surveys, 0), "Records in current filter")
        with kpi_row_1[1]:
            kpi_card("Total Yield", format_num(total_yield), "Tons (sum)")
        st.markdown('<div class="kpi-row-gap"></div>', unsafe_allow_html=True)
        with kpi_row_2[0]:
            kpi_card("Average Yield", format_num(avg_yield), "Tons per farm")
        with kpi_row_2[1]:
            kpi_card("Average Soil pH", format_num(avg_ph, 2), "Soil health indicator")

        adoption_rate, valid_seed_df, adopters_df = seed_adoption_donut(filtered_df)
        crop_legend = crop_distribution_legend(adopters_df)
        fill_segments = adoption_crop_segments(adoption_rate, adopters_df)
        legend_html = "".join(
            f'<span class="legend-chip"><span class="legend-dot" style="--dot:{entry["color"]};"></span>{entry["crop"]} {entry["share_pct"]:.0f}%</span>'
            for entry in crop_legend
        )
        donut_col, text_col = st.columns([1, 1.4], gap="large")
        with donut_col:
            donut_bg = str(THEME["secondary_bg"])
            donut_border = str(THEME["line"])
            donut_value_color = str(THEME["text"])
            donut_label_color = str(THEME["muted"])
            donut_track = str(THEME["donut_track"])
            donut_component_id = f"donut-{uuid.uuid4().hex[:10]}"

            donut_html = f"""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
                #{donut_component_id}.donut-root {{
                    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    background: var(--donut-bg, {donut_bg});
                    border: 1px solid var(--donut-border, {donut_border});
                    border-radius: 12px;
                    min-height: 214px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    position: relative;
                    overflow: hidden;
                }}
                #{donut_component_id} .donut-value {{
                    font-family: 'Space Grotesk', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    font-size: 1.95rem;
                    line-height: 1;
                    color: var(--donut-value, {donut_value_color});
                    letter-spacing: 0.01em;
                }}
                #{donut_component_id} .donut-label {{
                    margin-top: 0.25rem;
                    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    font-size: 0.95rem;
                    color: var(--donut-label, {donut_label_color});
                    font-weight: 600;
                }}
                #{donut_component_id} .donut-seg {{
                    fill: none;
                    stroke-width: 20;
                    stroke-linecap: butt;
                    stroke-dasharray: 0 100;
                }}
            </style>
            <div id="{donut_component_id}" class="donut-root">
                <div style="position:relative; width:248px; height:150px;">
                    <svg viewBox="0 0 248 150" role="img" aria-label="Seed adoption gauge" style="width:248px;height:150px;display:block;">
                        <path d="M 20 130 A 104 104 0 0 1 228 130" pathLength="100"
                              fill="none" stroke="{donut_track}" stroke-width="20" stroke-linecap="butt"></path>
                        {"".join(
                            f'<path class="donut-seg" d="M 20 130 A 104 104 0 0 1 228 130" pathLength="100" '
                            f'stroke="{seg["color"]}" stroke-dashoffset="-{seg["start"]:.4f}" '
                            f'data-final="{seg["length"]:.4f} 100" data-delay="{int(float(seg["start"]) * 10)}"></path>'
                            for seg in fill_segments
                        )}
                    </svg>
                    <div style="
                        position:absolute;
                        left:50%;
                        bottom:8px;
                        transform:translateX(-50%);
                        text-align:center;
                    ">
                        <div class="donut-value">{adoption_rate:.1f}%</div>
                        <div class="donut-label">Seed Adoption</div>
                    </div>
                </div>
            </div>
            <script>
            (() => {{
                const root = document.getElementById("{donut_component_id}");
                if (!root) return;

                const setThemeFromParent = () => {{
                    try {{
                        const parentDoc = window.parent?.document;
                        if (!parentDoc) return;
                        const appContainer =
                            parentDoc.querySelector('[data-testid="stAppViewContainer"]') ||
                            parentDoc.body;
                        const appStyles = window.parent.getComputedStyle(appContainer);
                        const bg = appStyles.backgroundColor || "";
                        const text = appStyles.color || "";
                        const m = bg.match(/\\d+/g);
                        let dark = {str(IS_DARK_THEME).lower()};
                        if (m && m.length >= 3) {{
                            const r = Number(m[0]);
                            const g = Number(m[1]);
                            const b = Number(m[2]);
                            const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
                            dark = luminance < 0.54;
                        }}
                        if (dark) {{
                            root.style.setProperty("--donut-bg", "{THEME['secondary_bg'] if IS_DARK_THEME else '#101a24'}");
                            root.style.setProperty("--donut-border", "rgba(127, 207, 159, 0.28)");
                            root.style.setProperty("--donut-value", text || "#e7ece9");
                            root.style.setProperty("--donut-label", "rgba(210, 224, 216, 0.72)");
                        }} else {{
                            root.style.setProperty("--donut-bg", "{THEME['secondary_bg'] if not IS_DARK_THEME else '#f3f7f4'}");
                            root.style.setProperty("--donut-border", "rgba(47, 106, 74, 0.22)");
                            root.style.setProperty("--donut-value", text || "#24312d");
                            root.style.setProperty("--donut-label", "rgba(64, 84, 73, 0.72)");
                        }}
                    }} catch (e) {{
                        // Keep server-provided defaults.
                    }}
                }};

                const animateFill = () => {{
                    if (root.dataset.animated === "1") return;
                    root.dataset.animated = "1";
                    const segments = root.querySelectorAll(".donut-seg");
                    segments.forEach((seg) => {{
                        const finalDash = seg.getAttribute("data-final") || "0 100";
                        const delay = Number(seg.getAttribute("data-delay") || "0");
                        seg.style.transition = `stroke-dasharray 640ms ease ${{delay}}ms`;
                        requestAnimationFrame(() => {{
                            seg.style.strokeDasharray = finalDash;
                        }});
                    }});
                }};

                setThemeFromParent();
                window.addEventListener("resize", setThemeFromParent);

                try {{
                    const frameEl = window.frameElement;
                    if (frameEl && window.parent?.IntersectionObserver) {{
                        const parentObserver = new window.parent.IntersectionObserver((entries) => {{
                            entries.forEach((entry) => {{
                                if (entry.isIntersecting) {{
                                    animateFill();
                                    parentObserver.disconnect();
                                }}
                            }});
                        }}, {{ threshold: 0.35 }});
                        parentObserver.observe(frameEl);
                    }} else {{
                        animateFill();
                    }}
                }} catch (e) {{
                    animateFill();
                }}
            }})();
            </script>
            """
            components.html(donut_html, height=224, scrolling=False)
        with text_col:
            st.markdown('<div class="adoption-copy">', unsafe_allow_html=True)
            st.markdown("### Adoption Snapshot")
            st.write(
                "The ring colors represent crop distribution among records with improved-seed adoption in the current filter."
            )
            st.write(
                "When district or crop filters change, segment sizes and legend shares update automatically."
            )
            if crop_legend:
                st.markdown(f'<div class="crop-legend">{legend_html}</div>', unsafe_allow_html=True)
            else:
                st.caption("No adoption records in the current filter.")
            st.markdown(
                '<p class="legend-note">Arc fill shows adoption rate. Legend colors indicate crop mix within adopting records.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if active_df is not None and not active_df.empty:
            st.markdown('<div class="panel"><h4>Data Contract Check</h4></div>', unsafe_allow_html=True)
            contract_rows = pd.DataFrame(data_contract_check(active_df))
            st.dataframe(contract_rows, width="stretch", hide_index=True)
            st.markdown('<div class="stack-gap"></div>', unsafe_allow_html=True)

        diff = data_diff_summary()
        if settings.get("show_data_diff", True) and diff:
            st.markdown('<div class="panel"><h4>Data Diff (Raw -> Clean -> Predictions)</h4></div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3, gap="large")
            with d1:
                kpi_card("Raw Columns", str(diff["raw_cols"]), f"Rows: {diff['raw_shape'][0]}")
            with d2:
                kpi_card("Clean Columns", str(diff["clean_cols"]), f"Rows: {diff['clean_shape'][0]}")
            with d3:
                kpi_card("Predictions Added", str(len(diff["added_by_prediction"])), "New output fields")

            with st.expander("See field-level changes", expanded=False):
                st.write(f"- Normalized/renamed fields: {len(diff['renamed_pairs'])}")
                st.write(f"- Removed after cleaning: {len(diff['removed_after_clean'])}")
                st.write(f"- Added by prediction: {len(diff['added_by_prediction'])}")

                if diff["renamed_pairs"]:
                    st.markdown("**Renamed Fields**")
                    st.dataframe(pd.DataFrame(diff["renamed_pairs"]).head(25), width="stretch")

                if diff["removed_after_clean"]:
                    st.markdown("**Removed/Absent After Cleaning**")
                    st.dataframe(
                        pd.DataFrame({"normalized_raw_column": diff["removed_after_clean"]}).head(25),
                        width="stretch",
                    )

                if diff["added_by_prediction"]:
                    st.markdown("**Prediction Output Columns**")
                    st.dataframe(
                        pd.DataFrame({"prediction_column": diff["added_by_prediction"]}),
                        width="stretch",
                    )
            st.markdown('<div class="stack-gap"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><h4>Yield Distribution</h4></div>', unsafe_allow_html=True)
        if yield_col:
            series = pd.to_numeric(filtered_df[yield_col], errors="coerce").dropna()
            if series.empty:
                st.warning("Yield values are not numeric in the current view.")
            else:
                display_series = series.copy()
                upper_cap = float(series.quantile(0.95))
                outlier_count = int((series > upper_cap).sum())
                if outlier_count > 0 and len(series) >= 8:
                    display_series = series.clip(upper=upper_cap)

                if len(display_series) >= 4:
                    bins = np.histogram_bin_edges(display_series, bins="fd")
                    if len(bins) < 5 or len(bins) > 16:
                        bins = int(settings.get("dashboard_hist_bins", 10))
                else:
                    bins = int(settings.get("dashboard_hist_bins", 8))
                if isinstance(bins, np.ndarray):
                    maxbins = int(max(5, min(40, len(bins) - 1)))
                else:
                    maxbins = int(bins)
                n_obs = int(len(display_series))
                v_min = float(display_series.min())
                v_max = float(display_series.max())
                v_range = max(v_max - v_min, 1e-9)
                use_step_bins = n_obs <= 25 and v_range > 0
                bin_step = None
                if use_step_bins:
                    target_bins = max(6, min(12, int(np.sqrt(n_obs) * 2)))
                    raw_step = max(0.5, v_range / target_bins)
                    magnitude = 10 ** np.floor(np.log10(raw_step))
                    candidates = np.array([1.0, 2.0, 2.5, 5.0, 10.0]) * magnitude
                    larger = candidates[candidates >= raw_step]
                    bin_step = float(larger[0] if len(larger) else candidates[-1])

                xlabel = "Yield (tons)"
                if outlier_count > 0 and len(series) >= 8:
                    xlabel = f"Yield (tons, display capped at {upper_cap:.1f})"

                chart_df = pd.DataFrame({"yield_value": display_series.to_numpy()})
                hist = (
                    alt.Chart(chart_df)
                    .mark_bar(
                        color="#2fbd66" if IS_DARK_THEME else "#2c9e5b",
                        opacity=0.9,
                        stroke="#1d7343" if IS_DARK_THEME else "#2a7e4a",
                        strokeWidth=1,
                    )
                    .encode(
                        x=alt.X(
                            "yield_value:Q",
                            bin=alt.Bin(step=bin_step) if use_step_bins else alt.Bin(maxbins=maxbins),
                            title=xlabel,
                        ),
                        y=alt.Y("count():Q", title="Count"),
                        tooltip=[alt.Tooltip("count():Q", title="Count")],
                    )
                    .properties(height=380)
                )
                rug = (
                    alt.Chart(chart_df)
                    .mark_tick(
                        color="#89f0b0" if IS_DARK_THEME else "#2a7e4a",
                        thickness=1.2,
                        size=10,
                        opacity=0.65,
                    )
                    .encode(
                        x=alt.X("yield_value:Q"),
                        y=alt.value(374),
                        tooltip=[alt.Tooltip("yield_value:Q", title="Yield", format=".2f")],
                    )
                )

                mean_rule = (
                    alt.Chart(pd.DataFrame({"x": [float(display_series.mean())]}))
                    .mark_rule(color="#f7b267", strokeDash=[8, 4], strokeWidth=2)
                    .encode(x="x:Q")
                )
                median_rule = (
                    alt.Chart(pd.DataFrame({"x": [float(display_series.median())]}))
                    .mark_rule(color="#7db7ff", strokeDash=[7, 3, 2, 3], strokeWidth=2)
                    .encode(x="x:Q")
                )

                chart = (
                    (hist + rug + mean_rule + median_rule)
                    .configure_view(stroke=None)
                    .configure_axis(gridOpacity=0.28)
                )

                st.altair_chart(chart, width="stretch")
                st.caption(
                    f"Mean {display_series.mean():.2f} t | Median {display_series.median():.2f} t"
                )
                if outlier_count > 0 and len(series) >= 8:
                    noun = "record" if outlier_count == 1 else "records"
                    verb = "was" if outlier_count == 1 else "were"
                    st.caption(
                        f"{outlier_count} high-value {noun} {verb} capped in chart display for readability; underlying data is unchanged."
                    )
        else:
            st.warning("No yield field detected in the current dataset.")

        chart_col = pick_geography_column(filtered_df)
        st.markdown('<div class="panel"><h4>Farmer Distribution</h4></div>', unsafe_allow_html=True)
        if chart_col:
            top_regions = (
                filtered_df[chart_col]
                .astype(str)
                .value_counts()
                .head(int(settings.get("top_regions_limit", 10)))
                .sort_values()
            )
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.barh(top_regions.index, top_regions.values, color="#f7b267")
            ax.set_xlabel("Survey Count")
            ax.grid(alpha=0.2, axis="x")
            st.pyplot(fig, width="stretch")
            st.caption(f"Grouping field: `{chart_col}`")
        else:
            st.info("Geographic grouping is unavailable for this dataset. Add a district/region/location-style column to enable this chart.")

        st.markdown('<div class="panel"><h4>Key Insights</h4></div>', unsafe_allow_html=True)
        insight_items = []
        if yield_col:
            q1 = filtered_df[yield_col].quantile(0.25)
            q3 = filtered_df[yield_col].quantile(0.75)
            insight_items.append(f"Interquartile yield range is {q1:.2f} to {q3:.2f}.")
        if ph_col:
            insight_items.append(f"Average soil pH in view is {avg_ph:.2f}.")
        if selected_district != "All":
            insight_items.append(f"Filters currently scoped to district: {selected_district}.")
        if selected_crop != "All":
            insight_items.append(f"Crop focus: {selected_crop}.")

        if not insight_items:
            insight_items = ["Add richer agronomic columns (yield, pH, rainfall) for stronger insight narratives."]

        for item in insight_items:
            st.write(f"- {item}")


# -----------------------------------------------------------------------------
# Page: Model Performance
# -----------------------------------------------------------------------------
elif page == "Model Performance":
    settings = app_settings()
    st.subheader("Model Performance")
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    metrics = load_metrics(str(settings["metrics_path"]))
    preds = load_csv(str(settings["predictions_path"]))

    if metrics:
        label_map = {
            "rmse": ("Typical error (tons)", "Large misses weigh more; lower is better"),
            "mae": ("Average error (tons)", "Average prediction miss; lower is better"),
            "n_train": ("Records used to learn", "Training examples"),
            "n_test": ("Records used to check", "Validation examples"),
            "test_size": ("Test share", "Fraction held out for checking"),
            "target": ("Prediction target", "What the model predicts"),
        }
        preferred_keys = ["rmse", "mae", "n_train", "n_test", "test_size", "target"]
        metric_keys = [k for k in preferred_keys if k in metrics]
        if not metric_keys:
            metric_keys = list(metrics.keys())
        top_keys = metric_keys[:4]
        if len(top_keys) >= 3:
            row_one = top_keys[:2]
            row_two = top_keys[2:4]
            cols_row_one = st.columns(2, gap="large")
            for i, key in enumerate(row_one):
                with cols_row_one[i]:
                    label, note = label_map.get(key, (key.upper(), ""))
                    kpi_card(label, format_num(metrics.get(key), 3), note)
            if row_two:
                st.markdown('<div class="kpi-row-gap"></div>', unsafe_allow_html=True)
                cols_row_two = st.columns(2, gap="large")
                for i, key in enumerate(row_two):
                    with cols_row_two[i]:
                        label, note = label_map.get(key, (key.upper(), ""))
                        kpi_card(label, format_num(metrics.get(key), 3), note)
        else:
            cols = st.columns(max(1, len(top_keys)), gap="large")
            for i, key in enumerate(top_keys):
                with cols[i]:
                    label, note = label_map.get(key, (key.upper(), ""))
                    kpi_card(label, format_num(metrics.get(key), 3), note)
        st.markdown('<div class="kpi-row-gap"></div>', unsafe_allow_html=True)

        readiness_label, readiness_text = model_readiness_message(metrics)
        st.info(f"Model status: {readiness_label}. {readiness_text}")
        st.caption(
            "Plain-language guide: if error numbers go down over time while records used to learn go up, your model is becoming more trustworthy."
        )
    else:
        st.info("Model metrics not found yet. Run `python scripts/train_model.py`.")

    with st.expander("Model Configuration", expanded=False):
        st.code(
            """
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

Preprocessing:
- Categorical: OneHotEncoder + SimpleImputer
- Numeric: SimpleImputer (median)
            """.strip(),
            language="python",
        )

    if preds is None or preds.empty:
        st.warning("No predictions available. Run `python scripts/predict.py`.")
    else:
        st.markdown('<div class="panel"><h4>Prediction Sample</h4></div>', unsafe_allow_html=True)
        st.dataframe(preds.head(15), width="stretch")

        actual_col = pick_column(preds, ["actual", "actual_yield", "yield_tons"])
        pred_col = pick_column(preds, ["prediction", "predicted", "predicted_yield"])
        if actual_col and pred_col:
            st.markdown('<div class="panel"><h4>Predicted vs Actual</h4></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7.4, 4.2))
            ax.scatter(preds[actual_col], preds[pred_col], alpha=0.55, color="#2b8a4b")
            lo = min(preds[actual_col].min(), preds[pred_col].min())
            hi = max(preds[actual_col].max(), preds[pred_col].max())
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="#9c2f2f", linewidth=1)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.grid(alpha=0.2)
            st.pyplot(fig, width="stretch")


# -----------------------------------------------------------------------------
# Page: Data Explorer
# -----------------------------------------------------------------------------
elif page == "Data Explorer":
    settings = app_settings()
    st.subheader("Data Explorer")
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Cleaned Data", "Features"])

    with tab1:
        raw = load_csv(str(settings["raw_data_path"]))
        if raw is None:
            raw = load_csv(str(settings["raw_fallback_path"]))
        if raw is None:
            st.warning(
                f"No raw data found at `{settings['raw_data_path']}` or `{settings['raw_fallback_path']}`."
            )
        else:
            st.caption(f"Rows: {raw.shape[0]} | Columns: {raw.shape[1]}")
            st.dataframe(raw.head(30), width="stretch")

    with tab2:
        clean = load_csv(str(settings["clean_data_path"]))
        if clean is None:
            st.warning(f"No cleaned data found at `{settings['clean_data_path']}`.")
        else:
            st.caption(f"Rows: {clean.shape[0]} | Columns: {clean.shape[1]}")
            st.dataframe(clean.head(30), width="stretch")

    with tab3:
        features = load_csv(str(settings["features_data_path"]))
        if features is None:
            st.warning(f"No feature set found at `{settings['features_data_path']}`.")
        else:
            st.caption(f"Rows: {features.shape[0]} | Features: {features.shape[1]}")
            st.dataframe(features.head(30), width="stretch")


# -----------------------------------------------------------------------------
# Page: Settings
# -----------------------------------------------------------------------------
elif page == "Settings":
    settings = app_settings()
    st.subheader("Settings and Configuration")
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    st.markdown("### App Settings")
    st.caption("These settings are saved to `data/processed/app_settings.json` and used on every rerun.")

    with st.form("app_settings_form", clear_on_submit=False):
        st.markdown("#### Data Paths")
        raw_data_path = st.text_input("Primary Raw Data Path", value=str(settings["raw_data_path"]))
        raw_fallback_path = st.text_input("Fallback Raw Data Path", value=str(settings["raw_fallback_path"]))
        clean_data_path = st.text_input("Clean Data Path", value=str(settings["clean_data_path"]))
        features_data_path = st.text_input("Features Data Path", value=str(settings["features_data_path"]))
        metrics_path = st.text_input("Metrics Path", value=str(settings["metrics_path"]))
        predictions_path = st.text_input("Predictions Path", value=str(settings["predictions_path"]))
        model_path = st.text_input("Model Artifact Path", value=str(settings["model_path"]))

        st.markdown("#### Dashboard Behavior")
        dashboard_hist_bins = st.slider(
            "Yield Histogram Bins",
            min_value=8,
            max_value=40,
            value=int(settings["dashboard_hist_bins"]),
            step=1,
        )
        top_regions_limit = st.slider(
            "Top Regions in Farmer Distribution",
            min_value=5,
            max_value=20,
            value=int(settings["top_regions_limit"]),
            step=1,
        )
        enable_animations = st.toggle("Enable Animations", value=bool(settings["enable_animations"]))
        show_data_diff = st.toggle("Show Data Diff Section", value=bool(settings["show_data_diff"]))

        submitted = st.form_submit_button("Save Settings")
        if submitted:
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
            st.success("Settings saved.")
            st.rerun()

    rc1, rc2 = st.columns(2, gap="large")
    with rc1:
        if st.button("Reset To Defaults", width="stretch"):
            st.session_state["app_settings"] = DEFAULT_SETTINGS.copy()
            save_app_settings(st.session_state["app_settings"])
            st.success("Settings reset to defaults.")
            st.rerun()
    with rc2:
        if st.button("Reload Settings From Disk", width="stretch"):
            st.session_state["app_settings"] = load_app_settings()
            st.success("Settings reloaded.")
            st.rerun()

    st.markdown("### Quick Start")
    st.code(
        """
# 1. Export Kobo data
cp <your-kobo-export>.csv data/raw/kobo_export.csv

# 2. Run full pipeline
python scripts/run_pipeline.py

# 3. Start dashboard
streamlit run streamlit_app.py
        """.strip(),
        language="bash",
    )


st.markdown('<div class="footer">2026 AgroSynapse | Building intelligence for global farming</div>', unsafe_allow_html=True)
