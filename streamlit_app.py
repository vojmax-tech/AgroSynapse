"""AgroSynapse Dashboard - Agricultural Survey Intelligence Platform."""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime
import re
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid
import warnings
from textwrap import dedent

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pandas.errors import ParserError

st.set_page_config(
    page_title="AgroSynapse | Agricultural Survey Intelligence",
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
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SIDEBAR_LOGO_PATH = ASSETS_DIR / "agrosynapse-sidebar-logo.svg"
_PROJECT_ROOT = Path(__file__).resolve().parent


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


def render_sidebar_brand() -> None:
    if SIDEBAR_LOGO_PATH.exists():
        try:
            logo_markup = SIDEBAR_LOGO_PATH.read_text(encoding="utf-8")
            st.sidebar.markdown(
                f'''
                <div class="sidebar-brand">
                    <div class="sidebar-brand-mark">{logo_markup}</div>
                    <div class="sidebar-brand-wordmark">AgroSynapse</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
            return
        except OSError:
            pass
    st.sidebar.markdown('<div class="sidebar-brand-fallback">AgroSynapse</div>', unsafe_allow_html=True)


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

PAGE_CANVAS = THEME["bg"] if IS_DARK_THEME else "#f5f7f5"
CARD_SURFACE = THEME["secondary_bg"] if IS_DARK_THEME else "#ffffff"
SOFT_SURFACE = "#151f2a" if IS_DARK_THEME else "#eef3ef"
SIDEBAR_SURFACE = "#0f1720" if IS_DARK_THEME else "#ffffff"
CONTROL_SURFACE = "#121c27" if IS_DARK_THEME else "#f8faf8"
CONTROL_BORDER = "rgba(127, 207, 159, 0.24)" if IS_DARK_THEME else "rgba(36, 49, 45, 0.12)"
CARD_SHADOW = "0 18px 40px rgba(0, 0, 0, 0.24)" if IS_DARK_THEME else "0 10px 28px rgba(15, 23, 42, 0.06)"
CARD_SHADOW_HOVER = "0 22px 44px rgba(0, 0, 0, 0.28)" if IS_DARK_THEME else "0 14px 32px rgba(15, 23, 42, 0.08)"

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
_STATIC_CSS_PATH = ASSETS_DIR / "theme_static.css"
if _STATIC_CSS_PATH.exists():
    st.markdown(
        f"<style>{_STATIC_CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <style>
    :root {{
        --bg-soft: {THEME["bg"]} !important;
        --card: {THEME["secondary_bg"]} !important;
        --ink: {THEME["text"]} !important;
        --muted: {THEME["muted"]} !important;
        --line: {THEME["line"]} !important;
        --surface-soft: {THEME["bg"]} !important;
        --surface-elevated: {THEME["secondary_bg"]} !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background:
            radial-gradient(circle at top right, rgba(127, 207, 159, 0.06), transparent 32%),
            linear-gradient(180deg, rgba(127, 207, 159, 0.035), transparent 28%),
            var(--bg-soft) !important;
    }}

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarUserContent"],
    [data-testid="stSidebarNav"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        background: var(--card) !important;
        background-color: var(--card) !important;
        opacity: 1 !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }}

    .page-hero,
    .dashboard-header {{
        background:
            radial-gradient(circle at 88% 18%, rgba(127, 207, 159, 0.26), transparent 28%),
            radial-gradient(circle at 12% 92%, rgba(255, 255, 255, 0.08), transparent 24%),
            linear-gradient(120deg, #173f27 0%, #1f6b3b 42%, #2b8a4b 100%);
        border: 1px solid rgba(212, 229, 216, 0.24);
        border-radius: 30px;
        padding: 2.35rem 2.55rem;
        margin-bottom: 1.7rem;
        box-shadow: 0 30px 56px rgba(10, 18, 28, 0.18);
        position: relative;
        overflow: hidden;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1.45rem;
        flex-wrap: wrap;
    }}

    .page-hero::before,
    .dashboard-header::before {{
        width: 240px;
        height: 240px;
        background: radial-gradient(circle, rgba(255,255,255,0.14) 0%, transparent 72%);
        transform: translate(42%, -46%);
    }}

    .page-hero .title-block,
    .dashboard-header .title-block {{
        max-width: min(46rem, 100%);
    }}

    .page-hero-eyebrow {{
        color: rgba(232, 245, 235, 0.82);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.74rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }}

    .page-hero h1,
    .dashboard-header h1 {{
        margin: 0;
        font-size: clamp(2rem, 2.7vw, 2.7rem);
        line-height: 0.98;
        color: white !important;
        letter-spacing: -0.015em;
    }}

    .page-hero-copy,
    .dashboard-subtitle {{
        margin: 1.05rem 0 0 0;
        color: rgba(232, 245, 235, 0.95);
        font-size: 1rem;
        line-height: 1.72;
        max-width: 47rem;
    }}

    .page-hero-badges {{
        display: flex;
        align-items: stretch;
        justify-content: flex-end;
        flex-wrap: wrap;
        gap: 0.9rem;
        position: relative;
        z-index: 1;
    }}

    .page-badge {{
        min-width: 164px;
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }}

    .page-badge-label {{
        display: block;
        color: rgba(232, 245, 235, 0.76);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.68rem;
        font-weight: 700;
        margin-bottom: 0.38rem;
    }}

    .page-badge-value {{
        display: block;
        color: white;
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.45;
    }}

    .kpi-card {{
        background:
            linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02) 44%, transparent 100%),
            var(--card);
        border: 1px solid rgba(127, 207, 159, 0.22);
        border-left: 4px solid var(--brand);
        border-radius: 24px;
        padding: 1.25rem 1.3rem 1.2rem 1.3rem;
        min-height: 150px;
        margin-bottom: 0.45rem;
        box-shadow: 0 16px 36px rgba(10, 18, 28, 0.08);
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        gap: 0.9rem;
        align-items: stretch;
    }}

    .kpi-card.no-icon {{
        gap: 0.8rem;
    }}

    .kpi-card-top {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        min-height: 48px;
    }}

    .kpi-card-top.no-icon {{
        min-height: auto;
    }}

    .kpi-card::before {{
        height: 0;
        display: none;
    }}

    .kpi-card::after {{
        content: '';
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: radial-gradient(circle at top right, rgba(255,255,255,0.08), transparent 30%);
    }}

    .kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 20px 42px rgba(10, 18, 28, 0.12);
        border-color: rgba(127, 207, 159, 0.34);
    }}

    .kpi-icon {{
        width: 48px;
        height: 48px;
        display: grid;
        place-items: center;
        border-radius: 15px;
        background: rgba(31, 107, 59, 0.1);
        border: 1px solid rgba(31, 107, 59, 0.14);
        color: var(--brand);
        font-size: 1rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
        flex: 0 0 48px;
    }}

    .kpi-card-meta {{
        display: grid;
        gap: 0.38rem;
        align-content: start;
        min-width: 0;
    }}

    .kpi-label {{
        color: var(--muted);
        font-size: 0.74rem;
        margin-bottom: 0;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        line-height: 1.24;
    }}

    .kpi-value {{
        color: var(--ink);
        font-size: clamp(1.6rem, 2.2vw, 2.05rem);
        font-weight: 800;
        margin-bottom: 0;
        line-height: 1.04;
        white-space: normal;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .kpi-delta {{
        color: var(--muted);
        font-size: 0.84rem;
        font-weight: 600;
        margin-top: auto;
        line-height: 1.58;
        min-height: 2.65em;
        display: flex;
        align-items: flex-end;
    }}

    .kpi-delta-empty {{
        visibility: hidden;
    }}

    .panel {{
        background: linear-gradient(180deg, rgba(255,255,255,0.045), transparent 100%), var(--card);
        border: 1px solid rgba(127, 207, 159, 0.18);
        border-radius: 24px;
        padding: 1.15rem 1.25rem;
        margin: 2.15rem 0 1.15rem;
        box-shadow: 0 12px 26px rgba(10, 18, 28, 0.05);
    }}

    .panel h4 {{
        font-size: 1.08rem;
        letter-spacing: 0.01em;
    }}

    .plain-note {{
        background: linear-gradient(180deg, rgba(127, 207, 159, 0.08), rgba(127, 207, 159, 0.04));
        border: 1px solid rgba(31, 107, 59, 0.16);
        border-radius: 22px;
        padding: 1.1rem 1.2rem;
        color: var(--ink);
        line-height: 1.7;
        margin-bottom: 1.2rem;
        max-width: none;
        box-shadow: 0 12px 24px rgba(10, 18, 28, 0.04);
    }}

    .smart-insight-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.045), transparent 100%), var(--card);
        border: 1px solid rgba(127, 207, 159, 0.18);
        border-radius: 24px;
        padding: 1.15rem 1.2rem 1.15rem 1.2rem;
        min-height: 196px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 14px 32px rgba(10, 18, 28, 0.06);
        margin-bottom: 0.6rem;
    }}

    .smart-insight-card::before {{
        content: '';
        position: absolute;
        inset: 0 0 auto 0;
        height: 4px;
        background: linear-gradient(90deg, #52d273 0%, #34c2a6 100%);
        opacity: 0.95;
    }}

    .smart-insight-card.warning::before {{
        background: linear-gradient(90deg, #f7b267 0%, #f39a58 100%);
    }}

    .smart-insight-card.neutral::before {{
        background: linear-gradient(90deg, #7db7ff 0%, #8479ff 100%);
    }}

    .smart-insight-card.accent::before {{
        background: linear-gradient(90deg, #d26cf3 0%, #8479ff 100%);
    }}

    .insight-kicker {{
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.7rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }}

    .insight-title {{
        font-family: 'Space Grotesk', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 1.18rem;
        font-weight: 700;
        color: var(--ink);
        line-height: 1.2;
        margin-bottom: 0.55rem;
        letter-spacing: -0.01em;
    }}

    .insight-body {{
        color: var(--ink);
        font-size: 0.94rem;
        line-height: 1.65;
    }}

    .insight-footnote {{
        margin-top: 0.8rem;
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.5;
    }}

    .smart-summary-title {{
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
        margin-bottom: 0.38rem;
    }}

    .smart-summary-body {{
        color: var(--ink);
        font-size: 0.96rem;
        line-height: 1.72;
    }}

    .stTabs [role="tablist"] {{
        gap: 0.65rem;
        margin-bottom: 0.95rem;
    }}

    .stTabs [role="tab"] {{
        border-radius: 999px !important;
        padding: 0.62rem 1.08rem !important;
        border: 1px solid rgba(127, 207, 159, 0.18) !important;
        background: rgba(127, 207, 159, 0.04) !important;
    }}

    .stTabs [role="tab"][aria-selected="true"] {{
        background: rgba(31, 107, 59, 0.12) !important;
        border-color: rgba(31, 107, 59, 0.24) !important;
    }}

    [data-testid="stExpander"] {{
        border: 1px solid rgba(127, 207, 159, 0.18) !important;
        border-radius: 20px !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.03), transparent 100%), var(--card) !important;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }}

    [data-testid="stExpander"] details {{
        border: none !important;
    }}


    .view-gap {{
        margin-top: 0.35rem;
        margin-bottom: 1.1rem;
    }}

    .section-gap {{
        margin-top: 1rem;
    }}

    .stack-gap {{
        height: 1.45rem;
    }}

    .kpi-row-gap {{
        height: 1rem;
    }}

    @media (max-width: 768px) {{
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] > div:first-child,
        div[data-testid="stSidebarContent"],
        div[data-testid="stSidebarUserContent"] {{
            background: var(--card) !important;
            background-color: var(--card) !important;
            opacity: 1 !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }}

        .page-hero,
        .dashboard-header {{
            padding: 1.5rem 1.3rem;
            border-radius: 25px;
            align-items: flex-start;
            gap: 1rem;
        }}

        .page-hero-badges {{
            width: 100%;
            justify-content: flex-start;
        }}

        .page-badge {{
            min-width: 132px;
            flex: 1 1 148px;
            padding: 0.95rem 1rem;
        }}

        .kpi-card {{
            min-height: 144px;
            padding: 1.1rem 1.05rem;
            gap: 0.82rem;
        }}

        .kpi-card-top {{
            gap: 0.72rem;
            min-height: 44px;
        }}

        .kpi-icon {{
            width: 44px;
            height: 44px;
            flex-basis: 44px;
        }}

        .smart-insight-card {{
            min-height: 0;
            padding: 1rem 1.05rem;
        }}

        .panel {{
            padding: 1rem 1.05rem;
        }}
    }}

    @media (max-width: 480px) {{
        .page-hero,
        .dashboard-header {{
            padding: 1.18rem 1rem;
        }}

        .kpi-card {{
            min-height: 138px;
            border-radius: 22px;
            padding: 1rem 0.95rem;
        }}

        .kpi-value {{
            font-size: 1.5rem;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <style>
    :root {{
        --page-bg: var(--background-color, {PAGE_CANVAS});
        --card-bg: var(--secondary-background-color, {CARD_SURFACE});
        --sidebar-bg: var(--secondary-background-color, {SIDEBAR_SURFACE});
        --soft-surface: {SOFT_SURFACE};
        --control-bg: {CONTROL_SURFACE};
        --control-border: {CONTROL_BORDER};
        --line-soft: {THEME["line"]};
        --ink: var(--text-color, {THEME["text"]});
        --muted: {THEME["muted"]};
        --card-shadow-strong: {CARD_SHADOW};
        --card-shadow-hover: {CARD_SHADOW_HOVER};
        --tab-shadow-active: {('0 8px 20px rgba(0, 0, 0, 0.18)' if IS_DARK_THEME else '0 8px 18px rgba(15, 23, 42, 0.06)')};
        --space-page-y: 1.1rem;
        --space-page-x: 1rem;
        --space-page-bottom: 2.1rem;
        --space-sidebar-y: 0.72rem;
        --space-sidebar-x: 0.6rem;
        --space-tight: 0.5rem;
        --space-regular: 0.95rem;
        --space-loose: 1.05rem;
    }}

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main {{
        background: var(--page-bg) !important;
        background-image: none !important;
    }}

    [data-testid="stMainBlockContainer"] {{
        padding-top: var(--space-page-y) !important;
        padding-left: var(--space-page-x) !important;
        padding-right: var(--space-page-x) !important;
        padding-bottom: var(--space-page-bottom) !important;
        max-width: none !important;
    }}

    h2, h3, h4 {{
        color: var(--ink) !important;
        margin-top: 0.9rem !important;
    }}

    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown p,
    .stMarkdown ul,
    .stMarkdown ol {{
        margin-bottom: 0.9rem !important;
    }}

    div[data-testid="column"] > div[data-testid="stVerticalBlock"] {{
        gap: 0.68rem !important;
    }}

    .stCaption {{
        margin-top: 0.15rem !important;
        margin-bottom: 0.55rem !important;
        color: var(--muted) !important;
    }}

    .sidebar-nav-title {{
        margin: 0.1rem 0 0.45rem 0.12rem;
        font-size: 0.66rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        font-weight: 700;
        color: rgba(231, 236, 233, 0.55);
    }}

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarUserContent"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    [data-testid="stSidebarNav"] {{
        background: var(--sidebar-bg) !important;
        background-color: var(--sidebar-bg) !important;
        background-image: none !important;
        opacity: 1 !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }}

    section[data-testid="stSidebar"] {{
        border-right: 1px solid var(--line-soft) !important;
        box-shadow: none !important;
    }}

    section[data-testid="stSidebar"][aria-expanded="true"] {{
        min-width: 252px !important;
        max-width: 252px !important;
    }}

    [data-testid="stSidebar"] .block-container {{
        padding: var(--space-sidebar-y) var(--space-sidebar-x) 1rem !important;
    }}

    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin: 0.08rem 0 0.9rem 0.08rem;
    }}

    .sidebar-brand-mark {{
        flex: 0 0 auto;
    }}

    .sidebar-brand svg {{
        display: block;
        width: 48px;
        height: auto;
    }}

    .sidebar-brand-wordmark {{
        color: #f5f8f6;
        font-size: 1.02rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1;
    }}

    .sidebar-brand-fallback {{
        margin: 0.15rem 0 0.85rem 0.08rem;
        color: var(--ink);
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    [data-testid="stSidebar"] .stRadio {{
        margin-bottom: 0.85rem !important;
    }}

    [data-testid="stSidebar"] .stRadio > label {{
        display: none !important;
    }}

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {{
        gap: 0.16rem !important;
    }}

    [data-testid="stSidebar"] label[data-baseweb="radio"] {{
        width: 100% !important;
        align-items: stretch !important;
        margin: 0 !important;
    }}

    [data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {{
        display: none !important;
    }}

    [data-testid="stSidebar"] label[data-baseweb="radio"] > div:last-child {{
        width: 100% !important;
    }}

    [data-testid="stSidebar"] label[data-baseweb="radio"] p {{
        width: 100% !important;
        margin: 0 !important;
        padding: 0.58rem 0.7rem 0.58rem 0.82rem !important;
        border-radius: 12px !important;
        border: 1px solid transparent !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        line-height: 1.18 !important;
        color: rgba(231, 236, 233, 0.76) !important;
        letter-spacing: -0.01em !important;
        text-align: left !important;
        transition: background 140ms ease, border-color 140ms ease, color 140ms ease !important;
    }}

    [data-testid="stSidebar"] label[data-baseweb="radio"]:hover p {{
        background: rgba(127, 207, 159, 0.07) !important;
        border-color: rgba(127, 207, 159, 0.12) !important;
        color: rgba(255, 255, 255, 0.96) !important;
    }}

    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p {{
        background: rgba(127, 207, 159, 0.12) !important;
        border-color: rgba(127, 207, 159, 0.22) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: inset 3px 0 0 rgba(82, 210, 115, 0.92) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] {{
        margin-top: 0.5rem !important;
        border-radius: 14px !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        padding: 0.7rem 0.82rem !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.88) !important;
    }}

    .stSidebar .stCaption,
    .stSidebar .stCaption span {{
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: rgba(231, 236, 233, 0.7) !important;
        letter-spacing: 0.01em !important;
    }}

    .page-topbar {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin: 0 0 var(--space-regular) 0;
    }}

    .page-topbar-title {{
        font-family: 'Space Grotesk', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        font-size: clamp(1.55rem, 2vw, 2rem);
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.02;
        color: var(--ink);
        margin: 0;
        white-space: nowrap;
    }}

    .page-topbar-rule {{
        flex: 1 1 auto;
        height: 1px;
        border-radius: 999px;
        background: var(--line-soft);
    }}

    .stTabs [role="tablist"] {{
        gap: 0.6rem !important;
        margin-bottom: 1rem !important;
    }}

    .stTabs [role="tab"] {{
        background: var(--soft-surface) !important;
        border: 1px solid var(--line-soft) !important;
        border-radius: 999px !important;
        color: var(--muted) !important;
        min-height: auto !important;
        padding: 0.5rem 0.88rem !important;
    }}

    .stTabs [role="tab"][aria-selected="true"] {{
        background: var(--card-bg) !important;
        color: var(--ink) !important;
        box-shadow: var(--tab-shadow-active) !important;
    }}

    .kpi-card {{
        background: var(--card-bg) !important;
        border: 1px solid var(--line-soft) !important;
        border-left: 1px solid var(--line-soft) !important;
        border-radius: 20px !important;
        padding: 0.96rem 1rem 0.92rem 1rem !important;
        min-height: 128px !important;
        margin-bottom: 0.42rem !important;
        gap: 0.66rem !important;
        box-shadow: var(--card-shadow-strong) !important;
        overflow: hidden;
    }}

    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent, var(--brand)), rgba(0,0,0,0) 72%);
        display: block !important;
    }}

    .kpi-card::after {{
        display: none !important;
    }}

    .kpi-card:hover {{
        transform: translateY(-2px) !important;
        box-shadow: var(--card-shadow-hover) !important;
        border-color: var(--line-soft) !important;
    }}

    .kpi-card-top {{
        display: flex !important;
        align-items: flex-start !important;
        justify-content: space-between !important;
        gap: 0.75rem !important;
        min-height: auto !important;
    }}

    .kpi-card.no-icon .kpi-card-top {{
        justify-content: flex-start !important;
    }}

    .kpi-icon {{
        order: 2;
        margin-left: auto;
        width: 34px !important;
        height: 34px !important;
        border-radius: 12px !important;
        flex: 0 0 34px !important;
        box-shadow: none !important;
    }}

    .kpi-card.no-icon .kpi-icon {{
        display: none !important;
    }}

    .kpi-label {{
        order: 1;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        line-height: 1.25 !important;
    }}

    .kpi-value {{
        font-size: clamp(1.65rem, 2.1vw, 2.2rem) !important;
        line-height: 0.98 !important;
        letter-spacing: -0.035em !important;
        margin: 0 !important;
    }}

    .kpi-delta {{
        font-size: 0.76rem !important;
        line-height: 1.4 !important;
        min-height: 1.1em !important;
        margin-top: auto !important;
        color: var(--muted) !important;
    }}

    .panel,
    .smart-insight-card,
    .plain-note,
    .donut-wrap {{
        background: var(--card-bg) !important;
        border: 1px solid var(--line-soft) !important;
        border-radius: 20px !important;
        box-shadow: var(--card-shadow-strong) !important;
    }}

    .panel {{
        padding: 0.82rem 1rem !important;
        margin: 1.08rem 0 0.86rem !important;
    }}

    .panel h4 {{
        margin: 0 !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }}

    .smart-insight-card {{
        min-height: 152px !important;
        padding: 1rem !important;
    }}

    .plain-note {{
        padding: 0.96rem 1rem !important;
    }}

    .soft-section-title {{
        font-size: 1rem !important;
        margin-bottom: 0.22rem !important;
        font-weight: 700 !important;
    }}

    .section-hint {{
        font-size: 0.84rem !important;
        margin-bottom: 0.76rem !important;
    }}

    .view-gap {{
        display: block !important;
        height: 0.62rem !important;
        margin: 0 !important;
    }}

    .section-gap {{
        display: block !important;
        height: 0.76rem !important;
        margin: 0 !important;
    }}

    .kpi-row-gap {{
        display: block !important;
        height: 0.54rem !important;
        margin: 0 !important;
    }}

    .stack-gap {{
        height: 1.04rem !important;
    }}

    .insight-row-gap {{
        height: 1.05rem !important;
    }}

    .upload-preview-card-gap {{
        height: 1.15rem !important;
    }}

    .crop-legend {{
        gap: 0.45rem 0.55rem !important;
        margin-top: 0.72rem !important;
    }}

    .legend-chip {{
        box-shadow: none !important;
    }}

    .smart-summary-title {{
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }}

    .smart-summary-body {{
        color: var(--ink);
        line-height: 1.6;
        font-size: 0.94rem;
    }}

    .kpi-value,
    .insight-title,
    .insight-body,
    .panel h4,
    .soft-section-title {{
        color: var(--ink) !important;
    }}

    .kpi-label,
    .kpi-delta,
    .section-hint,
    .insight-kicker,
    .insight-footnote,
    .smart-summary-title {{
        color: var(--muted) !important;
    }}

    .stButton > button {{
        background: var(--card-bg) !important;
        color: var(--ink) !important;
        border: 1px solid var(--control-border) !important;
        border-radius: 999px !important;
        padding: 0.66rem 1rem !important;
        box-shadow: none !important;
    }}

    .stButton > button:hover {{
        border-color: rgba(31, 107, 59, 0.45) !important;
        box-shadow: none !important;
    }}

    .stButton > button[kind="primary"] {{
        background: #1f6b3b !important;
        color: white !important;
        border-color: #1f6b3b !important;
    }}

    .stFileUploader > div > div > div > div,
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stTextArea > div > div {{
        background: var(--control-bg) !important;
        border-color: var(--control-border) !important;
        border-radius: 14px !important;
    }}

    .stSelectbox,
    .stTextInput,
    .stNumberInput,
    .stTextArea,
    .stFileUploader,
    .stToggle,
    .stSlider,
    .stRadio,
    .stMultiSelect {{
        margin-bottom: 0.65rem !important;
    }}

    div[data-testid="stForm"] {{
        background: var(--card-bg) !important;
        border: 1px solid var(--line-soft) !important;
        border-radius: 20px !important;
        padding: 1rem 1.05rem !important;
        box-shadow: var(--card-shadow-strong) !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--card-bg) !important;
        border: 1px solid var(--line-soft) !important;
        border-radius: 20px !important;
        padding: 1.05rem !important;
        box-shadow: var(--card-shadow-strong) !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {{
        gap: 1.05rem !important;
    }}

    .upload-section-heading {{
        margin: 0.25rem 0 1.35rem !important;
        max-width: 52rem;
    }}

    .upload-section-heading.secondary {{
        margin-top: 0.35rem !important;
    }}

    .upload-section-kicker {{
        color: var(--brand) !important;
        font-size: 0.72rem !important;
        font-weight: 800 !important;
        letter-spacing: 0 !important;
        line-height: 1.2 !important;
        margin-bottom: 0.34rem !important;
        text-transform: uppercase !important;
    }}

    .upload-section-heading h2 {{
        color: var(--ink) !important;
        font-family: 'Space Grotesk', 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        font-size: clamp(1.55rem, 2.1vw, 2rem) !important;
        font-weight: 800 !important;
        letter-spacing: 0 !important;
        line-height: 1.05 !important;
        margin: 0 !important;
    }}

    .upload-section-heading p {{
        color: var(--muted) !important;
        font-size: 0.94rem !important;
        line-height: 1.62 !important;
        margin: 0.55rem 0 0 !important;
        max-width: 42rem !important;
    }}

    .upload-panel-heading {{
        align-items: flex-start;
        display: flex;
        gap: 0.78rem;
        margin-bottom: 0.55rem !important;
    }}

    .upload-panel-heading.compact {{
        margin-bottom: 1.15rem !important;
        margin-top: 0.12rem !important;
    }}

    .upload-panel-step {{
        align-items: center;
        background: rgba(127, 207, 159, 0.12);
        border: 1px solid rgba(127, 207, 159, 0.22);
        border-radius: 999px;
        color: var(--brand);
        display: inline-flex;
        flex: 0 0 auto;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0;
        line-height: 1;
        min-height: 1.65rem;
        padding: 0.48rem 0.66rem;
        text-transform: uppercase;
    }}

    .upload-subsection-label {{
        align-items: center;
        color: var(--muted) !important;
        display: flex;
        font-size: 0.74rem !important;
        font-weight: 800 !important;
        letter-spacing: 0 !important;
        line-height: 1.2 !important;
        margin: 1.08rem 0 0.52rem !important;
        text-transform: uppercase !important;
    }}

    .upload-subsection-label::after {{
        background: var(--line-soft);
        content: '';
        display: block;
        flex: 1 1 auto;
        height: 1px;
        margin-left: 0.72rem;
    }}

    .upload-panel-divider {{
        border-top: 1px solid var(--line-soft);
        margin: 1.35rem 0 0.95rem;
    }}

    .upload-panel-title {{
        color: var(--ink) !important;
        font-size: 1.06rem !important;
        font-weight: 800 !important;
        letter-spacing: 0 !important;
        line-height: 1.18 !important;
        margin-bottom: 0.18rem !important;
    }}

    .upload-panel-note {{
        color: var(--muted) !important;
        font-size: 0.86rem !important;
        line-height: 1.7 !important;
        margin-bottom: 0 !important;
    }}

    div[data-testid="stFileUploaderDropzone"] {{
        align-items: center !important;
        background: rgba(127, 207, 159, 0.075) !important;
        border: 2px dashed rgba(127, 207, 159, 0.48) !important;
        border-radius: 18px !important;
        min-height: 11rem !important;
        padding: 1.35rem 1.45rem !important;
        transition: background 0.18s ease, border-color 0.18s ease !important;
    }}

    div[data-testid="stFileUploaderDropzone"]:hover {{
        background: rgba(127, 207, 159, 0.11) !important;
        border-color: rgba(127, 207, 159, 0.72) !important;
    }}

    div[data-testid="stFileUploaderDropzone"] > div {{
        min-width: 0 !important;
    }}

    div[data-testid="stFileUploaderDropzone"] p {{
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        line-height: 1.42 !important;
        margin: 0 !important;
        overflow-wrap: anywhere !important;
    }}

    [data-testid="stExpander"] {{
        background: var(--card-bg) !important;
        border: 1px solid var(--line-soft) !important;
        border-radius: 18px !important;
        overflow: hidden !important;
        margin: 0.78rem 0 1.05rem !important;
    }}

    [data-testid="stExpander"] summary {{
        padding: 0.82rem 0.95rem !important;
    }}

    .stDataFrame,
    div[data-testid="stExpander"],
    [data-testid="stAlert"] {{
        border-radius: 16px !important;
    }}

    .stDataFrame,
    [data-testid="stAlert"] {{
        margin: 0.55rem 0 0.95rem !important;
    }}

    .footer {{
        background: transparent !important;
        border-top: 1px solid var(--line-soft) !important;
        margin-top: 1.5rem !important;
        padding-top: 1rem !important;
    }}

    @media (max-width: 768px) {{
        [data-testid="stMainBlockContainer"] {{
            padding-left: 0.95rem !important;
            padding-right: 0.95rem !important;
            padding-top: 0.86rem !important;
            padding-bottom: 1.45rem !important;
        }}

        section[data-testid="stSidebar"][aria-expanded="true"] {{
            min-width: 232px !important;
            max-width: 232px !important;
        }}

        [data-testid="stSidebar"] .block-container {{
            padding: 0.72rem 0.62rem 0.82rem !important;
        }}

        .page-topbar {{
            margin-bottom: 0.7rem !important;
        }}

        .stTabs [role="tablist"] {{
            gap: 0.45rem !important;
            margin-bottom: 0.72rem !important;
        }}

        .kpi-card {{
            min-height: 116px !important;
            padding: 0.92rem !important;
        }}

        .smart-insight-card {{
            min-height: 142px !important;
        }}

        .panel {{
            padding: 0.78rem 0.92rem !important;
            margin: 0.82rem 0 0.66rem !important;
        }}

        .view-gap {{
            height: 0.5rem !important;
        }}

        .stack-gap {{
            height: 0.82rem !important;
        }}

        .insight-row-gap,
        .upload-preview-card-gap {{
            height: 0.82rem !important;
        }}

        div[data-testid="stForm"] {{
            padding: 0.88rem 0.92rem !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            padding: 0.82rem !important;
        }}

        .upload-section-heading {{
            margin-bottom: 0.95rem !important;
        }}

        .upload-section-heading h2 {{
            font-size: 1.42rem !important;
        }}

        .upload-section-heading p {{
            font-size: 0.88rem !important;
        }}

        .upload-panel-heading {{
            gap: 0.62rem !important;
        }}

        .upload-panel-heading.compact {{
            margin-bottom: 0.92rem !important;
        }}

        .upload-panel-step {{
            font-size: 0.62rem !important;
            min-height: 1.48rem !important;
            padding: 0.42rem 0.55rem !important;
        }}

        .upload-panel-title {{
            font-size: 1rem !important;
        }}

        .upload-subsection-label {{
            margin-top: 0.86rem !important;
        }}

        div[data-testid="stFileUploaderDropzone"] {{
            min-height: 9rem !important;
            padding: 1rem !important;
        }}
    }}

    @media (max-width: 480px) {{
        [data-testid="stMainBlockContainer"] {{
            padding-left: 0.72rem !important;
            padding-right: 0.72rem !important;
            padding-top: 0.78rem !important;
        }}

        section[data-testid="stSidebar"][aria-expanded="true"] {{
            min-width: 236px !important;
            max-width: 236px !important;
        }}

        [data-testid="stSidebar"] .block-container {{
            padding: 0.74rem 0.62rem 0.85rem !important;
        }}

        .page-topbar-title {{
            white-space: normal;
            line-height: 1.05;
        }}

        .view-gap {{
            height: 0.42rem !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            padding: 0.68rem !important;
        }}

        .upload-panel-heading {{
            display: grid !important;
            gap: 0.52rem !important;
        }}

        .upload-panel-step {{
            justify-self: start !important;
        }}

        .upload-panel-heading.compact {{
            margin-bottom: 0.78rem !important;
        }}

        .upload-subsection-label::after {{
            margin-left: 0.5rem !important;
        }}

        div[data-testid="stFileUploaderDropzone"] {{
            min-height: 8rem !important;
        }}
    }}
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
@st.cache_data(show_spinner=False)
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

    # Clear old CSV data across the data folder after a successful upload
    data_root = Path("data")
    if data_root.exists():
        for old_csv in data_root.rglob("*.csv"):
            try:
                if old_csv.resolve() != target.resolve():
                    old_csv.unlink()
            except OSError:
                pass

    return True, f"Saved `{target}` ({parsed.shape[0]} rows, {parsed.shape[1]} columns)."


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
    clean = re.sub(r"_+", " ", str(name)).strip()
    return clean.title() if clean else "Field"


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


def render_page_header(
    title: str,
    subtitle: str = "",
    chips: list[tuple[str, str]] | None = None,
    eyebrow: str = "",
) -> None:
    header_html = (
        '<div class="page-topbar">'
        f'<h1 class="page-topbar-title">{html.escape(title)}</h1>'
        '<div class="page-topbar-rule"></div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

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


def render_insight_cards(insights: list[dict[str, str]], per_row: int = 2, gap: str = "small") -> None:
    cleaned = [item for item in insights if item]
    if not cleaned:
        return

    width = max(1, min(per_row, 2 if len(cleaned) >= 4 else per_row))
    for start in range(0, len(cleaned), width):
        row = cleaned[start : start + width]
        columns = st.columns(len(row), gap=gap)
        for col, item in zip(columns, row):
            with col:
                st.markdown(
                    insight_card_html(
                        title=item.get("title", ""),
                        body=item.get("body", ""),
                        kicker=item.get("kicker", ""),
                        tone=item.get("tone", "neutral"),
                        footnote=item.get("footnote", ""),
                    ),
                    unsafe_allow_html=True,
                )
        if start + width < len(cleaned):
            st.markdown('<div class="insight-row-gap"></div>', unsafe_allow_html=True)


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


def compose_dashboard_brief(
    df: pd.DataFrame,
    metric_col: str | None,
    date_col: str | None = None,
    comparison_col: str | None = None,
) -> str:
    if df is None or df.empty:
        return "Once a dataset is available, AgroSynapse will automatically infer its strongest metrics, comparison axes, and data risks."

    completeness = float(df.notna().sum().sum()) / float(max(df.shape[0] * max(df.shape[1], 1), 1)) * 100.0
    parts: list[str] = [f"AgroSynapse is reading {len(df):,} visible records in this slice."]
    if metric_col:
        metric_series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
        if not metric_series.empty:
            parts.append(
                f"It is treating {pretty_column_label(metric_col)} as the lead measure, with a current average of {format_num(metric_series.mean(), 2)}."
            )
    if comparison_col:
        parts.append(f"{pretty_column_label(comparison_col)} is the strongest comparison axis in the current view.")
    if date_col:
        parts.append(f"{pretty_column_label(date_col)} provides a usable time signal for trend tracking.")
    if completeness >= 85:
        parts.append(f"Coverage is strong at {completeness:.1f}%, so the insights below are well-supported.")
    else:
        parts.append(f"Coverage is {completeness:.1f}%, so treat the weaker signals as directional rather than definitive.")

    return " ".join(parts)


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
    rows: list[dict[str, str]] = []
    if df is None or df.empty:
        return [
            {
                "Field": "Records",
                "Status": "Missing",
                "Mapped Column": "0 rows",
                "Priority": "Required",
            }
        ]

    def add_row(field: str, found: bool, mapped: str, priority: str) -> None:
        rows.append(
            {
                "Field": field,
                "Status": "Found" if found else "Missing",
                "Mapped Column": mapped if mapped else "-",
                "Priority": priority,
            }
        )

    add_row("Records", True, f"{len(df)} rows", "Required")

    target_col = pick_column(df, ["yield_tons", "yield", "yield_kg", "actual_yield_tons", "value", "target"])
    add_row("Target/Outcome", bool(target_col), target_col or "-", "Required")

    numeric_col = pick_numeric_column(
        df,
        preferred=["yield_tons", "yield", "value", "amount", "score", "prediction", "predicted_yield_tons"],
        exclude=[target_col] if target_col else [],
        min_valid_ratio=0.15,
    )
    numeric_ratio = (numeric_valid_ratio(df, numeric_col) * 100.0) if numeric_col else 0.0
    numeric_mapped = f"{numeric_col} ({numeric_ratio:.0f}% numeric)" if numeric_col else "-"
    add_row("Numeric Measure", bool(numeric_col), numeric_mapped, "Required")

    category_col = pick_category_column(
        df,
        exclude=[target_col, numeric_col] if target_col and numeric_col else ([target_col] if target_col else ([numeric_col] if numeric_col else [])),
        preferred_tokens=["crop", "type", "category", "indicator", "state", "region", "country", "district"],
    )
    if category_col:
        distinct = int(df[category_col].dropna().astype(str).nunique())
        add_row("Category Dimension", True, f"{category_col} ({distinct} classes)", "Required")
    else:
        add_row("Category Dimension", False, "-", "Required")

    geo_col = pick_geography_column(df)
    add_row("District/Region", bool(geo_col), geo_col or "-", "Recommended")

    seed_col = pick_column(df, ["improved_seed_used", "improved_seed", "seed_improved"])
    if seed_col:
        add_row("Seed Adoption", True, seed_col, "Optional")

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
        add_row("Time Dimension", True, f"{best_date_col} ({ratio * 100:.0f}% parseable)", "Recommended")
    else:
        add_row("Time Dimension", False, "-", "Recommended")

    total_cells = int(df.shape[0] * max(df.shape[1], 1))
    filled_cells = int(df.notna().sum().sum())
    completeness = (filled_cells / total_cells * 100.0) if total_cells else 0.0
    add_row("Data Completeness", completeness >= 60.0, f"{completeness:.1f}% non-missing", "Optional")

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


def detect_adoption_metrics(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Detect available adoption/practice metrics in the dataset.
    
    Returns list of (column_name, display_label, metric_type) tuples.
    """
    adoption_patterns = [
        (["improved_seed_used", "improved_seed", "seed_improved"], "Improved Seed"),
        (["irrigation_used", "irrigation", "uses_irrigation"], "Irrigation Usage"),
        (["fertilizer_used", "uses_fertilizer", "fertilizer"], "Fertilizer Usage"),
        (["pesticide_used", "uses_pesticide", "pesticide"], "Pesticide Usage"),
        (["organic_fertilizer", "uses_organic"], "Organic Inputs"),
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
    baseline_metrics = metrics.get("baseline_metrics")

    if isinstance(baseline_metrics, dict) and baseline_metrics.get("mae") is not None and mae is not None:
        try:
            if float(mae) > float(baseline_metrics["mae"]):
                return (
                    "Needs more signal",
                    "The learned model is not beating a simple median baseline yet. Collect more records before relying on predictions.",
                )
        except (TypeError, ValueError):
            pass

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


def model_display_label(model_type: object) -> str:
    labels = {
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
    }
    return labels.get(str(model_type), pretty_column_label(str(model_type))) if model_type else "Model"


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
NAV_PAGES = ["Dashboard", "Overview", "Model Performance", "Data Explorer", "Uploads", "Settings"]
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Dashboard"

if st.session_state["active_page"] not in NAV_PAGES:
    st.session_state["active_page"] = "Dashboard"

render_sidebar_brand()
st.sidebar.markdown('<div class="sidebar-nav-title"></div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigation",
    NAV_PAGES,
    key="active_page",
    label_visibility="collapsed",
)

active_df = get_active_frame()
district_filter_key = "global_district_filter"
crop_filter_key = "global_crop_filter"
district_filter_choice_key = "global_district_filter_choice"
crop_filter_choice_key = "global_crop_filter_choice"
dashboard_filters_open_key = "dashboard_filters_open"
secondary_filter_label = "Crop"
secondary_filter_col: str | None = None

if dashboard_filters_open_key not in st.session_state:
    st.session_state[dashboard_filters_open_key] = False


def reset_global_filters() -> None:
    st.session_state[district_filter_key] = "All"
    st.session_state[crop_filter_key] = "All"
    st.session_state[district_filter_choice_key] = "All"
    st.session_state[crop_filter_choice_key] = "All"


def toggle_dashboard_filters() -> None:
    st.session_state[dashboard_filters_open_key] = not st.session_state[dashboard_filters_open_key]

# Global Filters moved to Dashboard top bar

st.sidebar.markdown('<div class="sidebar-nav-title"></div>', unsafe_allow_html=True)
with st.sidebar.expander("Pipeline Status", expanded=False):
    for item in pipeline_status_items():
        st.write(f"- {item}")
    st.caption(last_run_badge())


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Page: Dashboard
# -----------------------------------------------------------------------------
if page == "Dashboard":
    settings = app_settings()
    dashboard_header_chips: list[tuple[str, str]] = [
        ("Pipeline", last_run_badge().replace("Last Run: ", "")),
    ]
    if active_df is not None and not active_df.empty:
        dashboard_header_chips.insert(0, ("Records", format_num(len(active_df), 0)))
        dashboard_header_chips.insert(1, ("Fields", format_num(active_df.shape[1], 0)))
        detected_target = infer_target_column(active_df)
        if detected_target:
            dashboard_header_chips.append(("Target candidate", pretty_column_label(detected_target)))
    else:
        dashboard_header_chips.insert(0, ("Dataset", "Awaiting upload"))

    render_page_header(
        "Dashboard",
        "A live operational view of your latest agricultural dataset with automatic pattern detection, benchmark gaps, and decision-focused data summaries.",
        chips=dashboard_header_chips,
        eyebrow="Operational intelligence",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    # Global Filter Bar
    if active_df is not None:
        district_col = pick_geography_column(active_df)
        crop_col = pick_column(active_df, ["crop", "crop_type", "main_crop"])
        secondary_filter_col = crop_col or pick_category_column(
            active_df,
            exclude=[district_col] if district_col else [],
            preferred_tokens=["crop", "type", "category", "indicator", "soil", "state", "region", "country"],
        )
        secondary_filter_label = "Crop" if crop_col else (
            pretty_column_label(secondary_filter_col) if secondary_filter_col else "Category"
        )

        selected_district = "All"
        selected_crop = "All"
        districts: list[str] = []
        crops: list[str] = []

        if district_col:
            districts = ["All"] + sorted(active_df[district_col].dropna().astype(str).unique().tolist())
            if st.session_state.get(district_filter_key) not in districts:
                st.session_state[district_filter_key] = "All"
            if st.session_state.get(district_filter_choice_key) not in districts:
                st.session_state[district_filter_choice_key] = st.session_state[district_filter_key]
            selected_district = st.session_state.get(district_filter_key, "All")

        if secondary_filter_col:
            crops = ["All"] + sorted(active_df[secondary_filter_col].dropna().astype(str).unique().tolist())
            if st.session_state.get(crop_filter_key) not in crops:
                st.session_state[crop_filter_key] = "All"
            if st.session_state.get(crop_filter_choice_key) not in crops:
                st.session_state[crop_filter_choice_key] = st.session_state[crop_filter_key]
            selected_crop = st.session_state.get(crop_filter_key, "All")

        filter_label = "Hide Filters ▲" if st.session_state[dashboard_filters_open_key] else "Show Filters ▼"
        filter_summary = f"Region: {selected_district}"
        if secondary_filter_col:
            filter_summary = f"{filter_summary} | {secondary_filter_label}: {selected_crop}"

        st.button(filter_label, key="dashboard_filter_toggle", on_click=toggle_dashboard_filters)
        st.caption(filter_summary)

        if st.session_state[dashboard_filters_open_key]:
            filter_cols = st.columns(2, gap="small")
            with filter_cols[0]:
                if district_col:
                    selected_district = st.pills(
                        "Region",
                        districts,
                        key=district_filter_choice_key,
                        width="stretch",
                    ) or "All"
                    st.session_state[district_filter_key] = selected_district
                else:
                    st.caption("No region field detected.")
            with filter_cols[1]:
                if secondary_filter_col:
                    selected_crop = st.pills(
                        secondary_filter_label,
                        crops,
                        key=crop_filter_choice_key,
                        width="stretch",
                    ) or "All"
                    st.session_state[crop_filter_key] = selected_crop
                else:
                    st.caption("No category field detected.")

    # Apply filters
    if active_df is not None:
        filtered_df = active_df.copy()
        district_col = pick_geography_column(filtered_df)
        crop_col = pick_column(filtered_df, ["crop", "crop_type", "main_crop"])
        active_secondary_col = crop_col or pick_category_column(
            filtered_df,
            exclude=[district_col] if district_col else [],
            preferred_tokens=["crop", "type", "category", "indicator", "soil", "state", "region", "country"],
        )

        if district_col and selected_district != "All":
            filtered_df = filtered_df[filtered_df[district_col].astype(str) == selected_district]
        if active_secondary_col and selected_crop != "All":
            filtered_df = filtered_df[filtered_df[active_secondary_col].astype(str) == selected_crop]
    else:
        filtered_df = None

    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    if filtered_df is None or filtered_df.empty:
        st.info("No survey dataset found yet. Add `data/raw/kobo_export.csv` or run the pipeline to unlock charts.")
    else:
        yield_col = pick_column(filtered_df, ["yield_tons", "yield", "yield_kg", "actual_yield_tons"])
        ph_col = pick_column(filtered_df, ["soil_ph", "ph", "soil_ph_level"])
        focus_numeric_col = yield_col or pick_numeric_column(
            filtered_df,
            preferred=["value", "target", "amount", "score", "prediction", "predicted_yield_tons"],
        )
        secondary_numeric_col = pick_numeric_column(
            filtered_df,
            preferred=["ph", "soil_ph", "soil_ph_level"],
            exclude=[focus_numeric_col] if focus_numeric_col else [],
        )

        focus_numeric_series = (
            pd.to_numeric(filtered_df[focus_numeric_col], errors="coerce").dropna()
            if focus_numeric_col
            else pd.Series(dtype="float64")
        )
        secondary_numeric_series = (
            pd.to_numeric(filtered_df[secondary_numeric_col], errors="coerce").dropna()
            if secondary_numeric_col
            else pd.Series(dtype="float64")
        )

        total_surveys = int(filtered_df.shape[0])
        total_cells = int(filtered_df.shape[0] * max(filtered_df.shape[1], 1))
        filled_cells = int(filtered_df.notna().sum().sum())
        completeness = (filled_cells / total_cells * 100.0) if total_cells else 0.0
        kpi_category_col = pick_category_column(
            filtered_df,
            exclude=[focus_numeric_col] if focus_numeric_col else [],
            preferred_tokens=["crop", "type", "category", "indicator", "state", "region", "country"],
        )
        kpi_category_unique = (
            int(filtered_df[kpi_category_col].dropna().astype(str).nunique())
            if kpi_category_col
            else None
        )

        focus_label = pretty_column_label(focus_numeric_col) if focus_numeric_col else "Primary Metric"
        focus_unit = "tons" if yield_col else "units"
        focus_total = float(focus_numeric_series.sum()) if not focus_numeric_series.empty else None
        focus_avg = float(focus_numeric_series.mean()) if not focus_numeric_series.empty else None
        secondary_avg = float(secondary_numeric_series.mean()) if not secondary_numeric_series.empty else None

        kpi_cols = st.columns(4, gap="small")
        with kpi_cols[0]:
            kpi_card(
                "Total Surveys",
                format_num(total_surveys, 0),
                "Current filter",
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M3 17h2v4H3zm4-6h2v10H7zm4-4h2v14h-2zm4-8h2v18h-2zm4 12h2v6h-2z"/></svg>',
                "#52d273",
            )
        with kpi_cols[1]:
            if focus_numeric_col and focus_total is not None:
                kpi_card(
                    f"Total {focus_label}",
                    format_num(focus_total),
                    "Across visible records",
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm2 14h10v-2H7v2zm0-4h8v-2H7v2zm0-4h6V7H7v2z"/></svg>',
                    "#34c2a6",
                )
            else:
                kpi_card(
                    "Data Columns",
                    format_num(filtered_df.shape[1], 0),
                    "Fields available",
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z"/></svg>',
                    "#49a9f1",
                )
        with kpi_cols[2]:
            if focus_numeric_col and focus_avg is not None:
                kpi_card(
                    f"Average {focus_label}",
                    format_num(focus_avg),
                    "Per visible record",
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 16l4-4 4 4 8-8"/><path d="M4 20h16"/></svg>',
                    "#8479ff",
                )
            else:
                kpi_card(
                    "Data Completeness",
                    f"{completeness:.1f}%",
                    "Non-missing cells",
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm-1 14.5l-3.5-3.5 1.41-1.41L11 13.67l5.59-5.59L18 9.5 11 16.5z"/></svg>',
                    "#d26cf3",
                )
        with kpi_cols[3]:
            if secondary_numeric_col and secondary_avg is not None:
                kpi_card(
                    f"Average {pretty_column_label(secondary_numeric_col)}",
                    format_num(secondary_avg, 2),
                    "Detected comparison metric",
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2a9 9 0 0 0-9 9v8a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-8a9 9 0 0 0-9-9zm-1 14h-2v-5h2zm4 0h-2v-3h2zm0-5h-2V7h2z"/></svg>',
                    "#f39a58",
                )
            elif kpi_category_col and kpi_category_unique is not None:
                kpi_card(
                    f"{pretty_column_label(kpi_category_col)} Classes",
                    format_num(kpi_category_unique, 0),
                    "Distinct groups in view",
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M5 4h14v3H5zm0 6h14v3H5zm0 6h14v3H5z"/></svg>',
                    "#f0627a",
                )
            else:
                kpi_card(
                    "Data Completeness",
                    f"{completeness:.1f}%",
                    "Non-missing cells",
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm-1 14.5l-3.5-3.5 1.41-1.41L11 13.67l5.59-5.59L18 9.5 11 16.5z"/></svg>',
                    "#d26cf3",
                )

        dashboard_date_col = pick_datetime_column(
            filtered_df,
            preferred=["date", "survey_date", "collection_date", "registration_date", "sample_date", "created_at", "timestamp"],
        )
        comparison_candidates: list[str] = []
        if district_col and selected_district == "All":
            comparison_candidates.append(district_col)
        if active_secondary_col and selected_crop == "All" and active_secondary_col not in comparison_candidates:
            comparison_candidates.append(active_secondary_col)
        if kpi_category_col and kpi_category_col not in comparison_candidates:
            comparison_candidates.append(kpi_category_col)

        dashboard_brief = compose_dashboard_brief(
            filtered_df,
            focus_numeric_col,
            date_col=dashboard_date_col,
            comparison_col=comparison_candidates[0] if comparison_candidates else None,
        )
        dashboard_insights = build_dashboard_insights(
            filtered_df,
            focus_numeric_col,
            comparison_candidates=comparison_candidates,
            date_col=dashboard_date_col,
            category_col=kpi_category_col,
        )

        st.markdown("### Insights")
        render_insight_cards(dashboard_insights, per_row=2)
        st.markdown('<div class="stack-gap"></div>', unsafe_allow_html=True)

        # Data Narrative Section
        st.markdown("### Trends")
        col_trend, col_bar = st.columns(2)
        
        with col_trend:
            st.markdown("#### Yield Trends Over Time")
            # Placeholder for line chart
            if focus_numeric_col:
                # Assume there's a date column
                date_col = pick_column(filtered_df, ["date", "survey_date", "collection_date"])
                if date_col:
                    chart_data = filtered_df[[date_col, focus_numeric_col]].dropna()
                    chart_data[date_col] = pd.to_datetime(chart_data[date_col], errors='coerce')
                    chart_data = chart_data.dropna().sort_values(date_col)
                    st.line_chart(chart_data.set_index(date_col)[focus_numeric_col])
                else:
                    st.info("No date column found for trend analysis.")
            else:
                st.info("No numeric column for trend analysis.")
        
        with col_bar:
            st.markdown("#### Regional Comparison")
            if district_col and focus_numeric_col:
                regional_avg = filtered_df.groupby(district_col)[focus_numeric_col].mean().dropna().sort_values(ascending=False).head(10)
                st.bar_chart(regional_avg)
            else:
                st.info("No region or numeric data for comparison.")

        seed_col_present = pick_column(filtered_df, ["improved_seed_used", "improved_seed", "seed_improved"]) is not None
        
        # Auto-detect best adoption metric to display
        adoption_metrics = detect_adoption_metrics(filtered_df)
        if adoption_metrics:
            # Use the first (highest priority) adoption metric found
            metric_col, metric_label, _ = adoption_metrics[0]
            adoption_rate, valid_seed_df, adopters_df = adoption_metric_donut(filtered_df, metric_col, metric_label)
        else:
            # Fallback to seed adoption for backwards compatibility
            adoption_rate, valid_seed_df, adopters_df = seed_adoption_donut(filtered_df)
            metric_label = "Seed Adoption"
        
        crop_legend = crop_distribution_legend(adopters_df)
        fill_segments = adoption_crop_segments(adoption_rate, adopters_df)
        legend_html = "".join(
            f'<span class="legend-chip"><span class="legend-dot" style="--dot:{entry["color"]};"></span>{entry["crop"]} {entry["share_pct"]:.0f}%</span>'
            for entry in crop_legend
        )

        snapshot_col = pick_category_column(
            filtered_df,
            exclude=[focus_numeric_col] if focus_numeric_col else [],
            preferred_tokens=["crop", "type", "category", "indicator", "state", "region", "country", "district"],
        )
        snapshot_share = (
            filtered_df[snapshot_col].dropna().astype(str).value_counts(normalize=True).head(7)
            if snapshot_col
            else pd.Series(dtype="float64")
        )

        donut_col, text_col = st.columns([1, 1.4], gap="large")
        if seed_col_present:
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
                        <svg viewBox="0 0 248 150" role="img" aria-label="{metric_label.lower()} gauge" style="width:248px;height:150px;display:block;">
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
                            <div class="donut-label">{metric_label}</div>
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
                st.markdown(f"### {metric_label} Snapshot")
                st.write(
                    f"The ring colors represent crop distribution among records with {metric_label.lower()} in the current filter."
                )
                st.write(
                    f"When district or {secondary_filter_label.lower()} filters change, segment sizes and legend shares update automatically."
                )
                if crop_legend:
                    st.markdown(f'<div class="crop-legend">{legend_html}</div>', unsafe_allow_html=True)
                else:
                    st.caption("No adoption records in the current filter.")
                st.markdown(
                    '<p class="legend-note">Arc fill shows adoption rate. Legend colors indicate crop mix within adopting records.</p>',
                    unsafe_allow_html=True,
                )
        else:
            with donut_col:
                st.markdown('<div class="panel"><h4>Category Snapshot</h4></div>', unsafe_allow_html=True)
                if snapshot_col and not snapshot_share.empty:
                    snapshot_df = pd.DataFrame(
                        {
                            "category": snapshot_share.index.tolist(),
                            "share": (snapshot_share.values * 100.0),
                        }
                    )
                    category_chart = (
                        alt.Chart(snapshot_df)
                        .mark_bar(color="#2b8a4b")
                        .encode(
                            x=alt.X("share:Q", title="Share (%)"),
                            y=alt.Y("category:N", sort="-x", title=pretty_column_label(snapshot_col)),
                            tooltip=[
                                alt.Tooltip("category:N", title=pretty_column_label(snapshot_col)),
                                alt.Tooltip("share:Q", title="Share (%)", format=".1f"),
                            ],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(category_chart, width="stretch")
                else:
                    st.info("No suitable category column found for a snapshot chart.")
            with text_col:
                st.write(
                    f"Smart mode is active for current CSV. Snapshot cards and charts are inferred from available columns ({filtered_df.shape[1]} fields)."
                )
                if snapshot_col and not snapshot_share.empty:
                    st.write(
                        f"Primary grouping field: `{snapshot_col}`. Top segment: `{snapshot_share.index[0]}` ({snapshot_share.iloc[0] * 100:.1f}%)."
                    )
                if focus_numeric_col and not focus_numeric_series.empty:
                    st.write(
                        f"Primary metric field: `{focus_numeric_col}`. Median value in current filter: {focus_numeric_series.median():.2f}."
                    )
                st.markdown(
                    '<p class="legend-note">When you switch datasets, the snapshot auto-picks the best available category and numeric fields.</p>',
                    unsafe_allow_html=True,
                )

        diff = data_diff_summary()
        show_contract = active_df is not None and not active_df.empty
        show_diff = bool(settings.get("show_data_diff", True) and diff)

        if show_contract or show_diff:
            st.markdown("### Data Quality")
            st.markdown(
                '<div class="section-hint">Contract checks and transformation diagnostics are grouped in tabs to keep the dashboard easier to scan.</div>',
                unsafe_allow_html=True,
            )

            if show_contract and show_diff:
                contract_tab, diff_tab = st.tabs(["Contract Check", "Raw -> Clean Diff"])
            elif show_contract:
                contract_tab = st.tabs(["Contract Check"])[0]
                diff_tab = None
            else:
                contract_tab = None
                diff_tab = st.tabs(["Raw -> Clean Diff"])[0]

            if contract_tab is not None:
                with contract_tab:
                    contract_rows = pd.DataFrame(data_contract_check(active_df))
                    st.dataframe(contract_rows, width="stretch", hide_index=True)

            if diff_tab is not None:
                with diff_tab:
                    d1, d2, d3 = st.columns(3, gap="medium")
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
        distribution_col = focus_numeric_col
        distribution_title = f"{pretty_column_label(distribution_col)} Distribution" if distribution_col else "Numeric Distribution"
        st.markdown(f'<div class="panel"><h4>{distribution_title}</h4></div>', unsafe_allow_html=True)
        if distribution_col:
            series = pd.to_numeric(filtered_df[distribution_col], errors="coerce").dropna()
            if series.empty:
                st.warning(f"Values in `{distribution_col}` are not numeric in the current view.")
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

                xlabel = pretty_column_label(distribution_col)
                if yield_col and distribution_col == yield_col:
                    xlabel = "Yield (tons)"
                if outlier_count > 0 and len(series) >= 8:
                    xlabel = f"{xlabel} (display capped at {upper_cap:.1f})"

                chart_df = pd.DataFrame({"metric_value": display_series.to_numpy()})
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
                            "metric_value:Q",
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
                        x=alt.X("metric_value:Q"),
                        y=alt.value(374),
                        tooltip=[alt.Tooltip("metric_value:Q", title=pretty_column_label(distribution_col), format=".2f")],
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
                unit_suffix = " t" if (yield_col and distribution_col == yield_col) else ""
                st.caption(
                    f"Mean {display_series.mean():.2f}{unit_suffix} | Median {display_series.median():.2f}{unit_suffix}"
                )
                if outlier_count > 0 and len(series) >= 8:
                    noun = "record" if outlier_count == 1 else "records"
                    verb = "was" if outlier_count == 1 else "were"
                    st.caption(
                        f"{outlier_count} high-value {noun} {verb} capped in chart display for readability; underlying data is unchanged."
                    )
        else:
            st.warning("No numeric field detected in the current dataset.")

        chart_col = pick_geography_column(filtered_df)
        fallback_group_col = pick_category_column(
            filtered_df,
            exclude=[distribution_col, snapshot_col] if distribution_col and snapshot_col else (
                [distribution_col] if distribution_col else ([snapshot_col] if snapshot_col else [])
            ),
            preferred_tokens=["district", "region", "state", "country", "crop", "type", "category", "indicator"],
        )
        group_col = chart_col or fallback_group_col
        group_title = "Farmer Distribution" if chart_col else "Category Distribution"
        st.markdown(f'<div class="panel"><h4>{group_title}</h4></div>', unsafe_allow_html=True)
        if group_col:
            top_regions = (
                filtered_df[group_col]
                .astype(str)
                .value_counts()
                .head(int(settings.get("top_regions_limit", 10)))
                .sort_values()
            )
            fig, ax = plt.subplots(figsize=(10, 4.8))
            fig.patch.set_facecolor(PAGE_CANVAS)
            ax.barh(top_regions.index, top_regions.values, color="#f7b267")
            ax.set_xlabel("Survey Count")
            ax.set_ylabel("")
            style_chart_axes(ax, dark=IS_DARK_THEME)
            ax.grid(alpha=0.22, axis="x")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            if chart_col:
                st.caption(f"Grouping field: `{group_col}`")
            else:
                st.caption(f"No geography field detected. Using `{group_col}` as the best available grouping field.")
        else:
            st.info("No suitable grouping field detected for this dataset.")

        st.markdown('<div class="panel"><h4>Key Insights</h4></div>', unsafe_allow_html=True)
        insight_items = []
        if distribution_col and not focus_numeric_series.empty:
            q1 = float(focus_numeric_series.quantile(0.25))
            q3 = float(focus_numeric_series.quantile(0.75))
            insight_items.append(
                f"Interquartile range for {pretty_column_label(distribution_col)} is {q1:.2f} to {q3:.2f}."
            )
        if secondary_numeric_col and not secondary_numeric_series.empty:
            insight_items.append(
                f"Average {pretty_column_label(secondary_numeric_col)} in view is {secondary_numeric_series.mean():.2f}."
            )
        if selected_district != "All":
            insight_items.append(f"Filters currently scoped to district: {selected_district}.")
        if selected_crop != "All":
            insight_items.append(f"{secondary_filter_label} focus: {selected_crop}.")

        if not insight_items:
            insight_items = [
                "Add at least one numeric measure and one category field to unlock richer automatic insights."
            ]

        for item in insight_items:
            st.write(f"- {item}")


# -----------------------------------------------------------------------------
# Page: Model Performance
# -----------------------------------------------------------------------------
elif page == "Model Performance":
    settings = app_settings()
    metrics = load_metrics(str(settings["metrics_path"]))
    preds = load_csv(str(settings["predictions_path"]))
    model_skipped = bool(metrics and str(metrics.get("status", "")).lower() == "skipped")

    model_header_chips: list[tuple[str, str]] = [("Pipeline", last_run_badge().replace("Last Run: ", ""))]
    if metrics:
        target_value = metrics.get("target")
        if target_value:
            model_header_chips.insert(0, ("Target", pretty_column_label(str(target_value))))
        if not model_skipped and metrics.get("n_train") is not None:
            model_header_chips.append(("Training rows", format_num(metrics.get("n_train"), 0)))
        if not model_skipped and metrics.get("model_type"):
            model_header_chips.append(("Model", model_display_label(metrics.get("model_type"))))
        if model_skipped:
            model_header_chips.append(("Status", "Training skipped"))
        else:
            readiness_label, _ = model_readiness_message(metrics)
            model_header_chips.append(("Readiness", readiness_label))
    else:
        model_header_chips.insert(0, ("Model", "Artifacts not found"))

    render_page_header(
        "Model Performance",
        "Track the current model's error profile, sample size, and prediction behavior with the same executive-style framing used on the dashboard.",
        chips=model_header_chips,
        eyebrow="Model intelligence",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    if metrics:
        if model_skipped:
            reason = str(metrics.get("reason", "Target column was unavailable for model training."))
            st.warning(f"Model training skipped for the latest pipeline run. {reason}")
            rows = metrics.get("n_rows")
            cols = metrics.get("n_columns")
            used_features = metrics.get("n_features")
            detail_parts: list[str] = []
            if rows is not None:
                detail_parts.append(f"rows: {rows}")
            if cols is not None:
                detail_parts.append(f"columns: {cols}")
            if used_features is not None:
                detail_parts.append(f"features used for transform: {used_features}")
            if detail_parts:
                st.caption(" | ".join(detail_parts))
            st.caption("Set a valid target in the Uploads page to train and generate fresh predictions.")
        else:
            label_map = {
                "rmse": ("Typical error (tons)", "Large misses weigh more; lower is better"),
                "mae": ("Average error (tons)", "Average prediction miss; lower is better"),
                "n_train": ("Records used to learn", "Training examples"),
                "n_test": ("Records used to check", "Validation examples"),
                "test_size": ("Test share", "Fraction held out for checking"),
                "target": ("Prediction target", "What the model predicts"),
            }
            metric_icon_map = {
                "rmse": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M13 2 3 14h7l-1 8 10-12h-7z"/></svg>',
                "mae": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 3a9 9 0 1 0 9 9 9 9 0 0 0-9-9zm1 13h-2v-2h2zm0-4h-2V7h2z"/></svg>',
                "n_train": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M16 11a4 4 0 1 0-4-4 4 4 0 0 0 4 4zm-8 0a3 3 0 1 0-3-3 3 3 0 0 0 3 3zm8 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4zm-8 1c-2.33 0-7 1.17-7 3.5V19h7v-1.5A4.68 4.68 0 0 1 10 14.6 8.55 8.55 0 0 0 8 14z"/></svg>',
                "n_test": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M4 4h16v2H4zm0 5h16v2H4zm0 5h10v2H4zm0 5h10v2H4z"/></svg>',
                "test_size": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M5 3h14a2 2 0 0 1 2 2v14l-4-3-4 3-4-3-4 3V5a2 2 0 0 1 2-2z"/></svg>',
                "target": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 14a4 4 0 1 1 4-4 4 4 0 0 1-4 4zm0-6a2 2 0 1 0 2 2 2 2 0 0 0-2-2z"/></svg>',
            }
            metric_color_map = {
                "rmse": "#f39a58",
                "mae": "#f7b267",
                "n_train": "#52d273",
                "n_test": "#49a9f1",
                "test_size": "#8479ff",
                "target": "#d26cf3",
            }
            preferred_keys = ["rmse", "mae", "n_train", "n_test", "test_size", "target"]
            metric_keys = [k for k in preferred_keys if k in metrics]
            if not metric_keys:
                metric_keys = list(metrics.keys())
            top_keys = metric_keys[:4]
            if len(top_keys) >= 3:
                row_one = top_keys[:2]
                row_two = top_keys[2:4]
                cols_row_one = st.columns(2, gap="small")
                for i, key in enumerate(row_one):
                    with cols_row_one[i]:
                        label, note = label_map.get(key, (key.upper(), ""))
                        kpi_card(label, format_num(metrics.get(key), 3), note, metric_icon_map.get(key, ""), metric_color_map.get(key, ""))
                if row_two:
                    st.markdown('<div class="kpi-row-gap"></div>', unsafe_allow_html=True)
                    cols_row_two = st.columns(2, gap="small")
                    for i, key in enumerate(row_two):
                        with cols_row_two[i]:
                            label, note = label_map.get(key, (key.upper(), ""))
                            kpi_card(label, format_num(metrics.get(key), 3), note, metric_icon_map.get(key, ""), metric_color_map.get(key, ""))
            else:
                cols = st.columns(max(1, len(top_keys)), gap="small")
                for i, key in enumerate(top_keys):
                    with cols[i]:
                        label, note = label_map.get(key, (key.upper(), ""))
                        kpi_card(label, format_num(metrics.get(key), 3), note, metric_icon_map.get(key, ""), metric_color_map.get(key, ""))
            st.markdown('<div class="kpi-row-gap"></div>', unsafe_allow_html=True)

            readiness_label, readiness_text = model_readiness_message(metrics)
            st.info(f"Model status: {readiness_label}. {readiness_text}")
            model_type = model_display_label(metrics.get("model_type"))
            baseline_metrics = metrics.get("baseline_metrics") if isinstance(metrics.get("baseline_metrics"), dict) else {}
            dataset_profile = metrics.get("dataset_profile") if isinstance(metrics.get("dataset_profile"), dict) else {}
            if model_type != "Model" or baseline_metrics or dataset_profile:
                model_diag_cols = st.columns(3, gap="medium")
                with model_diag_cols[0]:
                    kpi_card(
                        "Selected Model",
                        model_type,
                        "Chosen from candidate training",
                        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M4 4h16v4H4zm0 6h7v10H4zm9 0h7v10h-7z"/></svg>',
                        "#52d273",
                    )
                with model_diag_cols[1]:
                    baseline_mae = baseline_metrics.get("mae") if baseline_metrics else None
                    kpi_card(
                        "Baseline MAE",
                        format_num(baseline_mae, 3),
                        "Median predictor benchmark",
                        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M3 11h18v2H3zm4-6h10v2H7zm0 12h10v2H7z"/></svg>',
                        "#f7b267",
                    )
                with model_diag_cols[2]:
                    encoded_features = dataset_profile.get("estimated_encoded_features") if dataset_profile else None
                    kpi_card(
                        "Encoded Features",
                        format_num(encoded_features, 0),
                        "Estimated after preprocessing",
                        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M4 5h16v2H4zm0 6h16v2H4zm0 6h10v2H4z"/></svg>',
                        "#49a9f1",
                    )
            profile_warnings = dataset_profile.get("warnings", []) if dataset_profile else []
            if profile_warnings:
                st.warning(" ".join(str(item) for item in profile_warnings))
            candidate_metrics = metrics.get("candidate_metrics")
            if isinstance(candidate_metrics, dict) and candidate_metrics:
                with st.expander("Model Candidate Comparison", expanded=False):
                    comparison_rows = []
                    for candidate_name, values in candidate_metrics.items():
                        if isinstance(values, dict):
                            comparison_rows.append(
                                {
                                    "Model": model_display_label(candidate_name),
                                    "MAE": values.get("mae"),
                                    "RMSE": values.get("rmse"),
                                    "R2": values.get("r2"),
                                }
                            )
                    if baseline_metrics:
                        comparison_rows.append(
                            {
                                "Model": "Median baseline",
                                "MAE": baseline_metrics.get("mae"),
                                "RMSE": baseline_metrics.get("rmse"),
                                "R2": baseline_metrics.get("r2"),
                            }
                        )
                    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
            st.caption(
                "Plain-language guide: if error numbers go down over time while records used to learn go up, your model is becoming more trustworthy."
            )
    else:
        st.info("Model metrics not found yet. Run `python scripts/train_model.py`.")

    with st.expander("Model Configuration", expanded=False):
        model_config_name = model_display_label(metrics.get("model_type")) if metrics and not model_skipped else "Random Forest"
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
            language="python",
        )

    if model_skipped:
        st.info("No new predictions were generated in the latest run because model training was skipped.")
        if preds is not None and not preds.empty:
            st.caption("Showing latest available predictions file, which may come from an earlier run.")

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
            fig.patch.set_facecolor(PAGE_CANVAS)
            ax.scatter(preds[actual_col], preds[pred_col], alpha=0.55, color="#2b8a4b")
            lo = min(preds[actual_col].min(), preds[pred_col].min())
            hi = max(preds[actual_col].max(), preds[pred_col].max())
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="#9c2f2f", linewidth=1)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            style_chart_axes(ax, dark=IS_DARK_THEME)
            ax.grid(alpha=0.22)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")


# -----------------------------------------------------------------------------
# Page: Data Explorer
# -----------------------------------------------------------------------------
elif page == "Data Explorer":
    settings = app_settings()
    explorer_chips = [
        ("Raw", "Available" if Path(str(settings["raw_data_path"])).exists() or Path(str(settings["raw_fallback_path"])).exists() else "Missing"),
        ("Clean", "Available" if Path(str(settings["clean_data_path"])).exists() else "Missing"),
        ("Features", "Available" if Path(str(settings["features_data_path"])).exists() else "Missing"),
    ]
    render_page_header(
        "Data Explorer",
        "Inspect raw, cleaned, and engineered datasets with the same polished shell used across the rest of the app.",
        chips=explorer_chips,
        eyebrow="Data inspection",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Cleaned Data", "Features"])

    with tab1:
        raw = load_csv(str(settings["raw_data_path"]))
        if raw is None:
            raw = load_csv(str(settings["raw_fallback_path"]))
        if raw is None:
            st.warning(
                f"No raw data found at `{settings['raw_data_path']}` or `{settings['raw_fallback_path']}`."
                " Upload a CSV on the Uploads page to get started."
            )
        else:
            dl_col, cap_col = st.columns([1, 4])
            with cap_col:
                st.caption(f"Rows: {raw.shape[0]} | Columns: {raw.shape[1]}")
            with dl_col:
                st.download_button(
                    "Download CSV",
                    raw.to_csv(index=False).encode("utf-8"),
                    file_name="raw_data.csv",
                    mime="text/csv",
                    key="dl_raw",
                )
            st.dataframe(raw, width="stretch", height=420)

    with tab2:
        clean = load_csv(str(settings["clean_data_path"]))
        if clean is None:
            st.warning(
                f"No cleaned data found at `{settings['clean_data_path']}`."
                " Run the pipeline on the Uploads page to generate it."
            )
        else:
            dl_col, cap_col = st.columns([1, 4])
            with cap_col:
                st.caption(f"Rows: {clean.shape[0]} | Columns: {clean.shape[1]}")
            with dl_col:
                st.download_button(
                    "Download CSV",
                    clean.to_csv(index=False).encode("utf-8"),
                    file_name="clean_data.csv",
                    mime="text/csv",
                    key="dl_clean",
                )
            st.dataframe(clean, width="stretch", height=420)

    with tab3:
        features = load_csv(str(settings["features_data_path"]))
        if features is None:
            st.warning(
                f"No feature set found at `{settings['features_data_path']}`."
                " Run the pipeline on the Uploads page to generate it."
            )
        else:
            dl_col, cap_col = st.columns([1, 4])
            with cap_col:
                st.caption(f"Rows: {features.shape[0]} | Features: {features.shape[1]}")
            with dl_col:
                st.download_button(
                    "Download CSV",
                    features.to_csv(index=False).encode("utf-8"),
                    file_name="features.csv",
                    mime="text/csv",
                    key="dl_features",
                )
            st.dataframe(features, width="stretch", height=420)


# -----------------------------------------------------------------------------
# Page: Uploads
# -----------------------------------------------------------------------------
elif page == "Uploads":
    settings = app_settings()
    upload_header_chips = [
        ("Pipeline", last_run_badge().replace("Last Run: ", "")),
        ("Input path", Path(str(settings["raw_fallback_path"])).name or str(settings["raw_fallback_path"])),
    ]
    render_page_header(
        "Uploads & Pipeline",
        "Bring in a new CSV, preview what the app can already infer from it, and run the full pipeline without leaving the browser.",
        chips=upload_header_chips,
        eyebrow="Data onboarding",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    pipeline_feedback = st.session_state.pop("pipeline_run_feedback", None)
    if pipeline_feedback:
        st.success(pipeline_feedback)
    pipeline_output = st.session_state.get("pipeline_run_output")
    if pipeline_output:
        with st.expander("Last Pipeline Log", expanded=False):
            st.code(pipeline_output, language="text")

    st.markdown(
        """
        <div class="upload-section-heading">
            <h2>Upload a New Dataset</h2>
            <p>Choose a CSV, confirm the target field, and refresh the analytics data from one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_df: pd.DataFrame | None = None
    inferred_target: str | None = None
    auto_detect_target = True
    with st.container(border=True):
        st.markdown(
            """
            <div class="upload-panel-heading">
                <div class="upload-panel-step">Step 1</div>
                <div>
                    <div class="upload-panel-title">Source Configuration</div>
                    <div class="upload-panel-note">Set the CSV path and target detection before starting a run.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="upload-subsection-label">File source</div>', unsafe_allow_html=True)
        source_file_col, source_path_col = st.columns([1.15, 0.85], gap="large", vertical_alignment="top")
        with source_file_col:
            uploaded_csv = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                accept_multiple_files=False,
                help="Upload any CSV and save it to the pipeline input path.",
            )
        with source_path_col:
            pipeline_input_path = st.text_input(
                "Pipeline Input CSV Path",
                value=str(settings["raw_fallback_path"]),
                help="This file is used as `--input` for `scripts/run_pipeline.py`.",
            )

        preview_df = load_csv(pipeline_input_path.strip())
        inferred_target = infer_target_column(preview_df)

        st.markdown('<div class="upload-subsection-label">Target settings</div>', unsafe_allow_html=True)
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
                        "Pipeline Target Column",
                        options=ordered_columns,
                        index=ordered_columns.index(default_target),
                        help="Choose the exact target column to train on.",
                    )
                else:
                    pipeline_target_column = st.text_input(
                        "Pipeline Target Column",
                        value=inferred_target or "yield_tons",
                        help="Set the regression target for training (for example: `yield_tons`, `price`, or `value`).",
                    )
        with save_setting_col:
            update_fallback_setting = st.toggle(
                "Update saved raw path",
                value=True,
                help="When enabled, Settings will remember this input path as the fallback raw data path.",
            )
            st.caption("Keep this on when this CSV should become the default upload source.")

        st.markdown('<div class="upload-panel-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="upload-panel-heading compact">
                <div class="upload-panel-step">Step 2</div>
                <div>
                    <div class="upload-panel-title">Pipeline Actions</div>
                    <div class="upload-panel-note">Save the file for review, or run the full refresh when the source looks right.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        upload_col, run_col = st.columns(2, gap="medium")
        with upload_col:
            if st.button("Save Uploaded CSV", width="stretch"):
                saved_ok, saved_msg = save_uploaded_csv(uploaded_csv, pipeline_input_path.strip())
                if saved_ok:
                    if update_fallback_setting:
                        updated_settings = app_settings().copy()
                        updated_settings["raw_fallback_path"] = pipeline_input_path.strip()
                        st.session_state["app_settings"] = updated_settings
                        save_app_settings(updated_settings)
                    st.success(saved_msg)
                else:
                    st.error(saved_msg)

        with run_col:
            if st.button("Run Pipeline Now", type="primary", width="stretch"):
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

                with st.spinner("Running pipeline. This can take a moment..."):
                    run_ok, run_output = run_pipeline_for_input(
                        run_input_path,
                        pipeline_target_column,
                        app_settings(),
                    )

                if run_ok:
                    st.session_state["pipeline_run_feedback"] = "Pipeline completed. Dashboard and explorer data have been refreshed."
                    st.session_state["pipeline_run_output"] = run_output
                    load_csv.clear()
                    load_metrics.clear()
                    st.rerun()
                else:
                    st.error("Pipeline run failed.")
                    if run_output:
                        st.code(run_output, language="text")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            """
            <div class="upload-panel-heading">
                <div class="upload-panel-step">Status</div>
                <div>
                    <div class="upload-panel-title">Readiness Overview</div>
                    <div class="upload-panel-note">A quick health check of the selected pipeline file.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        status, status_msg = target_health_message(
            preview_df,
            pipeline_target_column,
            inferred_target=inferred_target,
            auto_detect=auto_detect_target,
        )
        if status == "good":
            st.success(status_msg)
        elif status == "warn":
            st.warning(status_msg)
        else:
            st.info(status_msg)

        summary_cols = st.columns(2, gap="medium")
        with summary_cols[0]:
            st.metric("Rows", format_num(preview_df.shape[0], 0) if preview_df is not None else "--")
        with summary_cols[1]:
            st.metric("Columns", format_num(preview_df.shape[1], 0) if preview_df is not None else "--")

        if preview_df is not None:
            st.caption(f"Detected dataset at path: {Path(pipeline_input_path.strip()).name or pipeline_input_path.strip()}")
            if inferred_target:
                st.caption(f"Suggested target from schema scan: `{inferred_target}`")
        else:
            st.caption("No dataset detected at the current path yet.")

    if preview_df is not None and not preview_df.empty:
        st.markdown('<div class="stack-gap"></div>', unsafe_allow_html=True)
        preview_overview = dataset_schema_overview(preview_df, target_candidate=inferred_target)
        readiness_score, readiness_band = dataset_readiness_score(preview_df, target_candidate=inferred_target)
        preview_insights = build_upload_preview_insights(preview_df, target_candidate=inferred_target)

        st.markdown(
            """
            <div class="upload-section-heading secondary">
                <div class="upload-section-kicker">Dataset preview</div>
                <h2>Dataset Intelligence Preview</h2>
                <p>Review readiness, target quality, and detected signals before using the data downstream.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        preview_stat_cols = st.columns(3, gap="medium")
        with preview_stat_cols[0]:
            readiness_color = "#52d273" if readiness_score >= 75 else "#f7b267" if readiness_score >= 55 else "#f39a58"
            kpi_card(
                "Readiness Score",
                f"{readiness_score}/100",
                readiness_band,
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2 3 7v6c0 5 3.8 9.74 9 11 5.2-1.26 9-6 9-11V7zm0 2.18 7 3.11V13c0 4.07-2.83 7.92-7 9.06C7.83 20.92 5 17.07 5 13V7.29z"/></svg>',
                readiness_color,
            )
        with preview_stat_cols[1]:
            target_display = pretty_column_label(preview_overview["target"]) if preview_overview["target"] else "Not detected"
            numeric_caption = f"{preview_overview['numeric_count']} numeric field(s)"
            kpi_card(
                "Best Target Candidate",
                target_display,
                numeric_caption,
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 14a4 4 0 1 1 4-4 4 4 0 0 1-4 4zm0-6a2 2 0 1 0 2 2 2 2 0 0 0-2-2z"/></svg>',
                "#49a9f1",
            )
        with preview_stat_cols[2]:
            kpi_card(
                "Detected Signals",
                f"{preview_overview['signals_count']} signals",
                preview_overview["signals_label"],
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M11 2v20a9 9 0 1 1 0-20zm2 0a9 9 0 0 1 0 20z"/></svg>',
                "#8479ff",
            )

        st.markdown('<div class="upload-preview-card-gap"></div>', unsafe_allow_html=True)
        render_insight_cards(preview_insights, per_row=2, gap="large")

        with st.expander("Preview rows", expanded=False):
            st.dataframe(preview_df.head(15), width="stretch")


# -----------------------------------------------------------------------------
# Page: Overview
# -----------------------------------------------------------------------------
elif page == "Overview":
    render_page_header("Overview")
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)
    
    st.markdown("## What is AgroSynapse?")
    render_markdown_block(
        """
        AgroSynapse is an intelligent platform that transforms agricultural survey data into actionable insights for farmers, researchers, and policymakers.

        **Key Capabilities:**
        - **Survey Data Ingestion**: Automatically process and clean raw survey data from KoboToolbox and other sources
        - **Predictive Modeling**: Train machine learning models to forecast crop yields and optimize farming practices
        - **Analytics Dashboard**: Visualize trends, adoption rates, and regional performance metrics
        - **Decision Support**: Provide data-driven recommendations for agricultural interventions
        """
    )
    
    st.markdown("## How It Works")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Data Flow")
        render_markdown_block(
            """
            1. **Ingest** raw survey data
            2. **Clean** and transform data
            3. **Train** predictive models
            4. **Analyze** and visualize results
            """
        )
    with col2:
        st.markdown("### Value Proposition")
        render_markdown_block(
            """
            - Reduce manual data processing time by 80%
            - Improve prediction accuracy with ML models
            - Enable evidence-based decision making
            - Scale agricultural monitoring efforts
            """
        )

# -----------------------------------------------------------------------------
# Page: Settings
# -----------------------------------------------------------------------------
elif page == "Settings":
    settings = app_settings()
    settings_score, settings_readiness_label, settings_recommendations = settings_readiness_summary(settings)
    settings_header_chips = [
        ("Readiness", settings_readiness_label),
        ("Animations", "On" if bool(settings["enable_animations"]) else "Off"),
        ("Data diff", "On" if bool(settings["show_data_diff"]) else "Off"),
        ("Histogram bins", str(int(settings["dashboard_hist_bins"]))),
    ]
    render_page_header(
        "Settings & Configuration",
        "Control paths, dashboard behavior, and quality-of-life options without changing the app's underlying pipeline logic.",
        chips=settings_header_chips,
        eyebrow="App controls",
    )
    st.markdown('<div class="view-gap"></div>', unsafe_allow_html=True)

    settings_health_rows = settings_path_health_rows(settings)
    ready_paths = sum(1 for row in settings_health_rows if row["Status"] == "Ready")
    attention_paths = len(settings_health_rows) - ready_paths

    smart_cols = st.columns(3, gap="medium")
    with smart_cols[0]:
        kpi_card(
            "Settings Readiness",
            f"{settings_score}/100",
            settings_readiness_label,
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2 3 7v6c0 5 3.8 9.74 9 11 5.2-1.26 9-6 9-11V7zm0 2.18 7 3.11V13c0 4.07-2.83 7.92-7 9.06C7.83 20.92 5 17.07 5 13V7.29z"/></svg>',
            "#52d273" if settings_score >= 85 else "#f7b267",
        )
    with smart_cols[1]:
        kpi_card(
            "Ready Paths",
            f"{ready_paths}/{len(settings_health_rows)}",
            "Existing and readable",
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M10 17 5 12l1.41-1.41L10 14.17l7.59-7.59L19 8z"/></svg>',
            "#34c2a6",
        )
    with smart_cols[2]:
        kpi_card(
            "Needs Attention",
            format_num(attention_paths, 0),
            "Missing or unreadable",
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M1 21h22L12 2zm12-3h-2v-2h2zm0-4h-2v-4h2z"/></svg>',
            "#f39a58" if attention_paths else "#52d273",
        )

    st.markdown('<div class="upload-preview-card-gap"></div>', unsafe_allow_html=True)
    settings_rec_col, settings_table_col = st.columns([0.85, 1.45], gap="large")
    with settings_rec_col:
        with st.container(border=True):
            st.markdown("#### Smart Recommendations")
            for recommendation in settings_recommendations:
                st.write(f"- {recommendation}")

            raw_primary_exists = Path(str(settings["raw_data_path"])).expanduser().exists()
            raw_fallback_exists = Path(str(settings["raw_fallback_path"])).expanduser().exists()
            if not raw_primary_exists and raw_fallback_exists:
                if st.button("Use Fallback CSV As Primary", width="stretch"):
                    updated_settings = settings.copy()
                    updated_settings["raw_data_path"] = str(settings["raw_fallback_path"])
                    st.session_state["app_settings"] = updated_settings
                    save_app_settings(updated_settings)
                    st.success("Primary raw path updated.")
                    st.rerun()

    with settings_table_col:
        with st.container(border=True):
            st.markdown("#### Path Health")
            st.dataframe(pd.DataFrame(settings_health_rows), width="stretch", hide_index=True)

    st.markdown('<div class="stack-gap"></div>', unsafe_allow_html=True)
    st.markdown("### App Settings")
    st.caption("These settings are saved to `data/processed/app_settings.json` and used on every rerun.")

    with st.form("app_settings_form", clear_on_submit=False):
        st.markdown("#### Data Paths")
        path_col_1, path_col_2 = st.columns(2, gap="large")
        with path_col_1:
            raw_data_path = st.text_input("Primary Raw Data Path", value=str(settings["raw_data_path"]))
            raw_fallback_path = st.text_input("Fallback Raw Data Path", value=str(settings["raw_fallback_path"]))
            clean_data_path = st.text_input("Clean Data Path", value=str(settings["clean_data_path"]))
            features_data_path = st.text_input("Features Data Path", value=str(settings["features_data_path"]))
        with path_col_2:
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
            load_csv.clear()
            load_metrics.clear()
            st.success("Settings reloaded.")
            st.rerun()

    st.markdown("### Quick Start")
    st.code(
        """
# 1. Option A (recommended): Upload CSV from Uploads page

# 1. Option B: Export Kobo data manually
cp <your-kobo-export>.csv data/raw/kobo_export.csv

# 2. Run full pipeline
python scripts/run_pipeline.py

# 3. Start dashboard
streamlit run streamlit_app.py
        """.strip(),
        language="bash",
    )


st.markdown('<div class="footer">2026 AgroSynapse | Building intelligence for global farming</div>', unsafe_allow_html=True)
