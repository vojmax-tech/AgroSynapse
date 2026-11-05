"""Tests for the training-time evaluation helpers.

The point of these is honesty of reporting rather than model quality: a forward-looking
split must not see future rows, and the naive benchmark it is compared against must be
built only from training-period data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_model import (  # noqa: E402
    detect_time_column,
    forest_capacity,
    persistence_baseline_metrics,
    temporal_holdout_metrics,
)


def make_panel(n_years: int = 12) -> pd.DataFrame:
    """Two series with different levels and a mild upward trend."""
    rows = []
    for series, (country, base) in enumerate({"Kenya": 1500.0, "Egypt": 6000.0}.items()):
        for offset in range(n_years):
            year = 2000 + offset
            rows.append(
                {
                    "country": country,
                    "crop": "Maize (corn)",
                    "year": year,
                    "area_harvested_ha": 1000.0 + 10 * offset + 100 * series,
                    "yield_kg_per_ha": base + 25.0 * offset,
                }
            )
    return pd.DataFrame(rows)


class TestForestCapacity:
    def test_larger_datasets_get_leafier_trees(self) -> None:
        small_trees, small_leaf = forest_capacity(200)
        large_trees, large_leaf = forest_capacity(25_000)
        assert large_leaf > small_leaf
        assert large_trees <= small_trees

    @pytest.mark.parametrize("n_rows", [0, 1, 999, 1_000, 5_000, 20_000, 10**7])
    def test_always_returns_usable_parameters(self, n_rows: int) -> None:
        n_estimators, min_samples_leaf = forest_capacity(n_rows)
        assert n_estimators >= 1 and min_samples_leaf >= 1


class TestDetectTimeColumn:
    def test_finds_year_column(self) -> None:
        assert detect_time_column(make_panel()) == "year"

    def test_ignores_year_with_too_few_periods(self) -> None:
        assert detect_time_column(make_panel(n_years=3)) is None

    def test_explicit_column_must_exist(self) -> None:
        panel = make_panel()
        assert detect_time_column(panel, "year") == "year"
        assert detect_time_column(panel, "not_a_column") is None


class TestTemporalHoldout:
    def test_train_and_test_periods_do_not_overlap(self) -> None:
        panel = make_panel(n_years=20)
        X = panel.drop(columns=["yield_kg_per_ha"])
        result = temporal_holdout_metrics(
            X,
            panel["yield_kg_per_ha"],
            "year",
            categorical_cols=["country", "crop"],
            numeric_cols=["year", "area_harvested_ha"],
            n_jobs=1,
            profile={"n_rows": len(X), "estimated_encoded_features": 10},
            model_kind="random_forest",
        )
        assert result is not None
        assert result["train_periods"][1] < result["test_periods"][0]
        assert result["n_train"] + result["n_test"] == len(panel)

    def test_returns_none_without_enough_periods(self) -> None:
        panel = make_panel(n_years=4)
        assert (
            temporal_holdout_metrics(
                panel.drop(columns=["yield_kg_per_ha"]),
                panel["yield_kg_per_ha"],
                "year",
                ["country", "crop"],
                ["year"],
                1,
                {"n_rows": len(panel), "estimated_encoded_features": 10},
                "random_forest",
            )
            is None
        )


class TestPersistenceBaseline:
    def test_carries_the_last_training_value_forward_per_series(self) -> None:
        panel = make_panel(n_years=10)
        train_mask = panel.year <= 2006
        test_mask = panel.year > 2006

        metrics = persistence_baseline_metrics(
            panel.drop(columns=["yield_kg_per_ha"]),
            panel["yield_kg_per_ha"],
            train_mask,
            test_mask,
            ["country", "crop"],
        )
        assert metrics is not None
        assert metrics["coverage"] == 1.0

        # Kenya's last training value is 1500 + 25*6; the trend adds 25/year afterwards,
        # so the errors on the three test years are 25, 50 and 75 for each series.
        assert metrics["mae"] == pytest.approx(50.0)

    def test_unseen_series_falls_back_without_raising(self) -> None:
        panel = make_panel(n_years=10)
        newcomer = panel.iloc[[0]].copy()
        newcomer["country"] = "Ghana"
        newcomer["year"] = 2009
        panel = pd.concat([panel, newcomer], ignore_index=True)

        train_mask = panel.year <= 2006
        test_mask = panel.year > 2006
        metrics = persistence_baseline_metrics(
            panel.drop(columns=["yield_kg_per_ha"]),
            panel["yield_kg_per_ha"],
            train_mask,
            test_mask,
            ["country", "crop"],
        )
        assert metrics is not None
        assert 0.0 < metrics["coverage"] < 1.0
        assert np.isfinite(metrics["mae"])

    def test_returns_none_without_categorical_grouping(self) -> None:
        panel = make_panel()
        assert (
            persistence_baseline_metrics(
                panel.drop(columns=["yield_kg_per_ha"]),
                panel["yield_kg_per_ha"],
                panel.year <= 2006,
                panel.year > 2006,
                [],
            )
            is None
        )
