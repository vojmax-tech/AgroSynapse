"""Shared feature policy for model training and feature transforms.

Three rules decide which columns reach the model:

1. **Risky columns** (:data:`RISKY_COLUMN_PATTERNS`) are always dropped - identifiers,
   names, phone numbers and raw dates carry personal data or row-order artefacts rather
   than agronomic signal.
2. **Leakage columns** (:data:`LEAKAGE_COLUMN_PATTERNS`) are always dropped from the
   feature set. They stay in the dataset because they are useful for reporting, but they
   are derived from the target and would produce meaningless "perfect" scores.
3. **The allowlist** (:data:`APPROVED_FEATURES`) names the columns of the shipped demo
   dataset. It is a default, not a hard gate: if the allowlist barely overlaps the
   dataset being trained on, :func:`resolve_feature_columns` falls back to every eligible
   column and reports that it did so. Without that fallback an allowlist written for a
   different schema silently trains the model on whatever one or two columns happened to
   match, which is indistinguishable from a working pipeline until you read the metrics.
"""

from __future__ import annotations

import re

# Columns of the shipped demo panel (data/raw/africa_crop_yield.csv), plus the
# smallholder-survey fields the KoboToolbox ingest path produces.
APPROVED_FEATURES = [
    # --- Africa crop-yield panel (FAOSTAT + World Bank) ---
    "country",
    "crop",
    "year",
    "area_harvested_ha",
    "fertilizer_kg_per_ha",
    "arable_land_pct",
    "agri_land_pct",
    "rural_population_pct",
    "agri_employment_pct",
    "gdp_per_capita_usd",
    "rural_electricity_pct",
    "avg_precipitation_mm",
    # Previous-season history. These are past observations, not the season being
    # predicted, so they are legitimate inputs rather than leakage - a forecaster in the
    # field knows last year's yield.
    "yield_prev_year_kg_per_ha",
    "yield_prev2_year_kg_per_ha",
    "yield_prev3_mean_kg_per_ha",
    "area_prev_year_ha",
    # --- KoboToolbox smallholder survey ---
    "sex",
    "age",
    "community_village",
    "district_lga",
    "state_region",
    "farm_size_ha",
    "soil_ph",
    "nitrogen_ppm",
    "phosphorus_ppm",
    "potassium_ppm",
    "organic_matter",
    "season",
    "soil_type",
    "uses_fertilizer",
    "fertilizer_type",
    "irrigation_used",
    "improved_seed_used",
    "extension_visits_last_12_months",
    "market_distance_km",
]

RISKY_COLUMN_PATTERNS = [
    r"(^|_)name($|_)",
    r"(^|_)full_name($|_)",
    r"(^|_)phone($|_)",
    r"(^|_)phone_number($|_)",
    r"(^|_)id($|_)",
    r"(^|_)farmer_id($|_)",
    r"(^|_)uuid($|_)",
    r"(^|_)iso3($|_)",
    r"(^|_)registration_date($|_)",
    r"(^|_)sample_date($|_)",
    r"(^|_)harvest_date($|_)",
    r"(^|_)date($|_)",
]

# Columns computable from the target. ``production_tonnes`` is exactly
# ``yield_kg_per_ha * area_harvested_ha / 1000``, so a model given both it and area
# recovers the target arithmetically and scores near-perfectly on nothing.
LEAKAGE_COLUMN_PATTERNS = [
    r"(^|_)production_tonnes($|_)",
    r"(^|_)production_kg($|_)",
    r"(^|_)total_production($|_)",
    r"(^|_)total_harvest($|_)",
]

# Below this share of eligible dataset columns, the allowlist is treated as written for a
# different schema and ignored in favour of every eligible column.
MIN_ALLOWLIST_COVERAGE = 0.5


def _matches_any(name: str, patterns: list[str]) -> bool:
    column = name.strip().lower()
    return any(re.search(pattern, column) for pattern in patterns)


def is_risky_column(name: str) -> bool:
    """True for identifier / personal-data / raw-date columns."""
    return _matches_any(name, RISKY_COLUMN_PATTERNS)


def is_leakage_column(name: str) -> bool:
    """True for columns derived from the prediction target."""
    return _matches_any(name, LEAKAGE_COLUMN_PATTERNS)


def eligible_columns(df_columns: list[str], target: str) -> list[str]:
    """Every column that is neither the target, risky, nor target-derived."""
    return [
        column
        for column in df_columns
        if column != target and not is_risky_column(column) and not is_leakage_column(column)
    ]


def select_allowlisted_features(
    df_columns: list[str], target: str, allowlist: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Intersect the dataset columns with the allowlist, dropping risky/leaky columns.

    Returns ``(selected, risky_in_allowlist, blocked_from_dataset)``.
    """
    allowlist_present = [c for c in allowlist if c in df_columns and c != target]
    risky_in_allowlist = [c for c in allowlist_present if is_risky_column(c) or is_leakage_column(c)]
    selected = [c for c in allowlist_present if c not in risky_in_allowlist]
    blocked_from_dataset = [
        column
        for column in df_columns
        if column not in selected
        and column != target
        and (is_risky_column(column) or is_leakage_column(column))
    ]
    return selected, risky_in_allowlist, blocked_from_dataset


def resolve_feature_columns(
    df_columns: list[str],
    target: str,
    allowlist: list[str],
    min_coverage: float = MIN_ALLOWLIST_COVERAGE,
) -> dict[str, object]:
    """Decide the final feature set, falling back when the allowlist does not fit.

    The returned dict carries the decision and its rationale so callers can log it and
    persist it alongside the model metrics:

    ``selected``
        Feature columns to train on.
    ``eligible``
        Columns that survived the risky/leakage filters.
    ``allowlist_matched``
        Allowlist columns actually present in the dataset.
    ``coverage``
        ``len(allowlist_matched) / len(eligible)`` - 1.0 when the allowlist fits fully.
    ``fallback_used`` / ``reason``
        Whether the allowlist was ignored, and why.
    ``blocked``
        Columns removed as risky or target-derived, with the reason for each.
    """
    eligible = eligible_columns(df_columns, target)
    matched, risky_in_allowlist, _ = select_allowlisted_features(df_columns, target, allowlist)

    coverage = len(matched) / len(eligible) if eligible else 0.0
    fallback_used = bool(eligible) and coverage < min_coverage

    if fallback_used:
        selected = eligible
        reason = (
            f"Allowlist matched {len(matched)} of {len(eligible)} eligible columns "
            f"({coverage:.0%}, below the {min_coverage:.0%} threshold). It was written for a "
            "different schema, so every eligible column is used instead."
        )
    else:
        selected = matched
        reason = f"Allowlist matched {len(matched)} of {len(eligible)} eligible columns ({coverage:.0%})."

    blocked = {
        column: ("target-derived" if is_leakage_column(column) else "identifier or personal data")
        for column in df_columns
        if column != target and (is_risky_column(column) or is_leakage_column(column))
    }

    return {
        "selected": selected,
        "eligible": eligible,
        "allowlist_matched": matched,
        "risky_in_allowlist": risky_in_allowlist,
        "coverage": round(coverage, 4),
        "fallback_used": fallback_used,
        "reason": reason,
        "blocked": blocked,
    }
