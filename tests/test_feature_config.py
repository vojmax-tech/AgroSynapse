"""Tests for the feature-selection policy.

The regression these cover: an allowlist written for one schema used to be intersected
with a different dataset, and whatever handful of columns happened to overlap became the
entire feature set. The pipeline reported success, and the only evidence was a negative
R-squared in the metrics file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from feature_config import (  # noqa: E402
    APPROVED_FEATURES,
    eligible_columns,
    is_leakage_column,
    is_risky_column,
    resolve_feature_columns,
)

PANEL_COLUMNS = [
    "country",
    "country_iso3",
    "crop",
    "year",
    "area_harvested_ha",
    "production_tonnes",
    "fertilizer_kg_per_ha",
    "arable_land_pct",
    "agri_land_pct",
    "rural_population_pct",
    "agri_employment_pct",
    "gdp_per_capita_usd",
    "rural_electricity_pct",
    "avg_precipitation_mm",
    "yield_kg_per_ha",
]
TARGET = "yield_kg_per_ha"


class TestColumnClassification:
    @pytest.mark.parametrize(
        "column", ["farm_id", "farmer_id", "full_name", "phone_number", "_uuid", "harvest_date", "country_iso3"]
    )
    def test_identifiers_are_risky(self, column: str) -> None:
        assert is_risky_column(column)

    @pytest.mark.parametrize("column", ["country", "crop", "year", "fertilizer_kg_per_ha", "soil_ph"])
    def test_feature_columns_are_not_risky(self, column: str) -> None:
        assert not is_risky_column(column)

    def test_production_is_treated_as_leakage(self) -> None:
        # production_tonnes == yield_kg_per_ha * area_harvested_ha / 1000
        assert is_leakage_column("production_tonnes")
        assert not is_leakage_column("area_harvested_ha")

    def test_eligible_excludes_target_identifiers_and_leakage(self) -> None:
        eligible = eligible_columns(PANEL_COLUMNS, TARGET)
        assert TARGET not in eligible
        assert "country_iso3" not in eligible
        assert "production_tonnes" not in eligible
        assert "country" in eligible and "crop" in eligible


class TestResolveFeatureColumns:
    def test_shipped_dataset_uses_every_eligible_column(self) -> None:
        decision = resolve_feature_columns(PANEL_COLUMNS, TARGET, APPROVED_FEATURES)
        assert decision["fallback_used"] is False
        assert set(decision["selected"]) == set(eligible_columns(PANEL_COLUMNS, TARGET))

    def test_mismatched_allowlist_falls_back_instead_of_training_on_one_column(self) -> None:
        """The exact failure this policy exists to prevent."""
        foreign_columns = [
            "farm_id",
            "crop_type",
            "farm_area_acres",
            "irrigation_type",
            "fertilizer_used_tons",
            "pesticide_used_kg",
            "soil_type",  # the single column that overlaps APPROVED_FEATURES
            "season",
            "water_usage_cubic_meters",
            "yield_tons",
        ]
        decision = resolve_feature_columns(foreign_columns, "yield_tons", APPROVED_FEATURES)

        assert decision["allowlist_matched"] == ["season", "soil_type"]
        assert decision["fallback_used"] is True
        assert "farm_area_acres" in decision["selected"]
        assert "water_usage_cubic_meters" in decision["selected"]
        assert len(decision["selected"]) > len(decision["allowlist_matched"])
        assert "different schema" in str(decision["reason"])

    def test_fallback_still_drops_risky_and_leaking_columns(self) -> None:
        columns = ["farm_id", "farmer_name", "production_tonnes", "rainfall_mm", "yield_kg_per_ha"]
        decision = resolve_feature_columns(columns, "yield_kg_per_ha", ["nothing_matches_here"])

        assert decision["fallback_used"] is True
        assert decision["selected"] == ["rainfall_mm"]
        assert decision["blocked"]["production_tonnes"] == "target-derived"
        assert decision["blocked"]["farm_id"] == "identifier or personal data"

    def test_matching_allowlist_is_respected(self) -> None:
        columns = ["country", "crop", "year", "unlisted_extra", "yield_kg_per_ha"]
        decision = resolve_feature_columns(columns, "yield_kg_per_ha", APPROVED_FEATURES)

        assert decision["fallback_used"] is False
        assert "unlisted_extra" not in decision["selected"]
        assert decision["coverage"] == pytest.approx(0.75)

    def test_dataset_of_only_blocked_columns_selects_nothing(self) -> None:
        decision = resolve_feature_columns(["farm_id", "_uuid", "yield_kg_per_ha"], "yield_kg_per_ha", APPROVED_FEATURES)
        assert decision["selected"] == []
