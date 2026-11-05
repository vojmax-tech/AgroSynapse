"""Tests for the FAOSTAT / World Bank dataset builder.

These cover the join key rather than the network calls. Country-name normalisation is
where the two sources silently disagree: FAOSTAT writes "Cote d'Ivoire" with an accent
and "United Republic of Tanzania", the World Bank writes "Tanzania" and "Egypt, Arab
Rep.". A mismatch does not raise - it just drops the country from an inner join.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_dataset import COLUMN_ORDER, COUNTRY_ALIASES, normalize_country  # noqa: E402


class TestNormalizeCountry:
    @pytest.mark.parametrize(
        ("faostat", "world_bank"),
        [
            ("Côte d'Ivoire", "Cote d'Ivoire"),
            ("Réunion", "Reunion"),
            ("Sao Tome and Principe", "São Tomé and Príncipe"),
            ("  Kenya  ", "Kenya"),
            ("Guinea-Bissau", "Guinea-Bissau"),
        ],
    )
    def test_accents_and_whitespace_fold_to_the_same_key(self, faostat: str, world_bank: str) -> None:
        assert normalize_country(faostat) == normalize_country(world_bank)

    def test_curly_and_straight_apostrophes_agree(self) -> None:
        assert normalize_country("Cote d’Ivoire") == normalize_country("Cote d'Ivoire")

    def test_distinct_countries_stay_distinct(self) -> None:
        assert normalize_country("Niger") != normalize_country("Nigeria")
        assert normalize_country("Congo") != normalize_country("Democratic Republic of the Congo")

    def test_mojibake_does_not_normalise_to_the_same_key(self) -> None:
        """Reading the UTF-8 FAOSTAT export as latin-1 produces this, and it must not match.

        The test documents why build_dataset.load_faostat pins encoding="utf-8": under
        latin-1 the country silently stops joining to World Bank rows.
        """
        mojibake = "Côte d'Ivoire".encode("utf-8").decode("latin-1")
        assert normalize_country(mojibake) != normalize_country("Cote d'Ivoire")


class TestAliases:
    @pytest.mark.parametrize("faostat_name", ["United Republic of Tanzania", "Egypt", "Gambia", "Somalia"])
    def test_alias_keys_are_normalised_form(self, faostat_name: str) -> None:
        assert normalize_country(faostat_name) in COUNTRY_ALIASES

    def test_alias_targets_are_already_normalised(self) -> None:
        for target in COUNTRY_ALIASES.values():
            assert normalize_country(target) == target


class TestPanelShape:
    def test_target_is_the_last_column_and_leakage_column_is_present(self) -> None:
        # production_tonnes ships for reporting; feature_config blocks it from the model.
        assert COLUMN_ORDER[-1] == "yield_kg_per_ha"
        assert "production_tonnes" in COLUMN_ORDER

    def test_committed_dataset_matches_the_builder_schema(self) -> None:
        dataset = Path(__file__).resolve().parents[1] / "data" / "raw" / "africa_crop_yield.csv"
        if not dataset.exists():
            pytest.skip("Demo dataset not present; run scripts/build_dataset.py")
        header = pd.read_csv(dataset, nrows=0).columns.tolist()
        assert header == COLUMN_ORDER
