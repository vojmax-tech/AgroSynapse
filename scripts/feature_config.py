"""Shared feature policy for model training and feature transforms."""

from __future__ import annotations

import re

APPROVED_FEATURES = [
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
    "season_e_g_2024_maize",
    "crop",
    "area_harvested_ha",
    "soil_type",
    "uses_fertilizer",
    "fertilizer_type",
    "fertilizer_type_organic",
    "fertilizer_type_urea",
    "fertilizer_type_npk",
    "fertilizer_type_compost",
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
    r"(^|_)registration_date($|_)",
    r"(^|_)sample_date($|_)",
    r"(^|_)harvest_date($|_)",
    r"(^|_)date($|_)",
]


def is_risky_column(name: str) -> bool:
    column = name.strip().lower()
    return any(re.search(pattern, column) for pattern in RISKY_COLUMN_PATTERNS)


def select_allowlisted_features(df_columns: list[str], target: str, allowlist: list[str]) -> tuple[list[str], list[str], list[str]]:
    allowlist_present = [col for col in allowlist if col in df_columns and col != target]
    risky_in_allowlist = [col for col in allowlist_present if is_risky_column(col)]
    selected = [col for col in allowlist_present if not is_risky_column(col)]
    blocked_from_dataset = [
        col for col in df_columns if col not in selected and col != target and is_risky_column(col)
    ]
    return selected, risky_in_allowlist, blocked_from_dataset
