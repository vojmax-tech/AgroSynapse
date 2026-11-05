"""Tests for the descriptive analysis module.

These pin down what the Data Quality and Crop Insights pages tell a reader: that real
problems are flagged, that clean data is not accused of problems it does not have, and
that every sentence in the written summary agrees with the numbers behind it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analysis import (  # noqa: E402
    benchmark_against_peers,
    compare_groups,
    group_difference_test,
    iita_crop_group,
    measure_label,
    measure_unit,
    period_values,
    quality_report,
    trend_by_group,
    write_summary,
)


def checks(report: dict) -> dict[tuple[str, str], dict]:
    return {(i["check"], i["column"]): i for i in report["issues"]}


def make_panel() -> pd.DataFrame:
    """Four countries x two crops x ten years, with known levels and trends."""
    rows = []
    levels = {"Nigeria": 0.8, "Ghana": 1.0, "Benin": 1.1, "Togo": 1.2}
    for country, factor in levels.items():
        for crop, base, growth in [("Cassava, fresh", 9000.0, -100.0), ("Maize (corn)", 1500.0, 40.0)]:
            for offset in range(10):
                rows.append(
                    {
                        "country": country,
                        "crop": crop,
                        "year": 2010 + offset,
                        "yield_kg_per_ha": factor * (base + growth * offset),
                    }
                )
    return pd.DataFrame(rows)


class TestUnitsAndLabels:
    @pytest.mark.parametrize(
        "column, unit, label",
        [
            ("yield_kg_per_ha", "kg/ha", "Yield"),
            ("yield_tons", "t", "Yield"),
            ("farm_size_ha", "ha", "Farm size"),
            ("rural_population_pct", "%", "Rural population"),
            ("price", "", "Price"),
        ],
    )
    def test_unit_and_label_from_column_name(self, column: str, unit: str, label: str) -> None:
        assert measure_unit(column) == unit
        assert measure_label(column) == label

    @pytest.mark.parametrize(
        "label, group",
        [
            ("Cassava, fresh", "Cassava"),
            ("Yams", "Yam"),
            ("Cow peas, dry", "Cowpea"),
            ("Soya beans", "Soybean"),
            ("Maize (corn)", "Maize"),
            ("Plantains and cooking bananas", "Banana & plantain"),
            ("Sweet potatoes", None),
            ("Sorghum", None),
            ("Groundnuts, excluding shelled", None),
        ],
    )
    def test_iita_mandate_crops_are_matched_by_word(self, label: str, group: str | None) -> None:
        assert iita_crop_group(label) == group

    def test_period_values_accepts_years_and_dates(self) -> None:
        assert period_values(pd.Series([2019, 2020])).tolist() == [2019, 2020]
        assert period_values(pd.Series(["2021-03-04", "2022-11-30"])).tolist() == [2021, 2022]


class TestQualityReport:
    def test_clean_panel_has_no_high_severity_issues(self) -> None:
        report = quality_report(make_panel(), key_columns=["country", "crop", "year"], group_column="crop")
        assert not [i for i in report["issues"] if i["severity"] == "high"]
        assert report["score"] >= 90

    def test_flags_the_problems_a_field_survey_actually_has(self) -> None:
        df = pd.DataFrame(
            {
                "farmer_id": range(1, 13),
                "crop": ["Maize", "maize ", "Maize", "Cassava"] * 3,
                "farm_size_ha": [1.2, -0.5, 2.0, 1.1, 0.9, 1.4, 1.0, 1.3, 0.8, 1.6, 1.2, 1.1],
                "yield_tons": ["2.1", "1.8", "2,4", "n/a", "2.0", "1.9", "2.2", "2.5", "1.7", "2.3", "2.0", "2.1"],
                "irrigation_pct": [10, 20, 150, 5, 0, 0, 30, 40, 10, 10, 20, 0],
                "survey_year": [2024] * 11 + [2204],
                "notes": [None] * 12,
            }
        )
        found = checks(quality_report(df))
        assert found[("Impossible negative values", "farm_size_ha")]["rows_affected"] == 1
        assert ("Numbers stored as text", "yield_tons") in found
        assert "n/a" in found[("Numbers stored as text", "yield_tons")]["examples"]
        assert ("Percentage above 100", "irrigation_pct") in found
        assert ("Implausible year", "survey_year") in found
        assert ("Empty column", "notes") in found
        assert ("Inconsistent labels", "crop") in found
        assert ("Extra spaces", "crop") in found

    def test_quantity_columns_named_after_a_year_are_not_years(self) -> None:
        df = make_panel().assign(yield_prev_year_kg_per_ha=lambda d: d["yield_kg_per_ha"])
        assert ("Implausible year", "yield_prev_year_kg_per_ha") not in checks(quality_report(df))

    def test_duplicates_and_repeated_keys(self) -> None:
        df = make_panel()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        found = checks(quality_report(df, key_columns=["country", "crop", "year"]))
        assert found[("Duplicate rows", "(all)")]["rows_affected"] == 1
        assert found[("Repeated records", "country + crop + year")]["rows_affected"] == 2

    def test_outliers_are_judged_within_their_crop(self) -> None:
        # Cassava out-yields maize several times over; that is agronomy, not an error.
        # A maize yield far above every other maize yield is the error.
        df = make_panel()
        assert not any(i["check"] == "Extreme values" for i in quality_report(df, group_column="crop")["issues"])
        df.loc[df.index[-1], "yield_kg_per_ha"] = 150_000.0
        found = checks(quality_report(df, group_column="crop"))
        assert found[("Extreme values", "yield_kg_per_ha")]["rows_affected"] == 1
        assert "Maize (corn)" in found[("Extreme values", "yield_kg_per_ha")]["examples"]

    def test_score_falls_with_severity(self) -> None:
        clean = quality_report(make_panel())["score"]
        broken = make_panel()
        broken.loc[0, "yield_kg_per_ha"] = -1
        assert quality_report(broken)["score"] < clean

    def test_empty_frame(self) -> None:
        report = quality_report(pd.DataFrame(columns=["a", "b"]))
        assert report["score"] == 0 and report["issues"][0]["check"] == "Empty dataset"


class TestComparisons:
    def test_confidence_interval_brackets_the_mean(self) -> None:
        table = compare_groups(make_panel(), "yield_kg_per_ha", "crop")
        assert list(table["group"]) == ["Cassava, fresh", "Maize (corn)"]
        assert (table["ci_low"] <= table["mean"]).all() and (table["mean"] <= table["ci_high"]).all()
        assert table["n"].tolist() == [40, 40]

    def test_small_groups_are_dropped(self) -> None:
        df = pd.DataFrame({"g": ["a", "a", "a", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
        assert compare_groups(df, "v", "g", min_n=3)["group"].tolist() == ["a"]

    def test_difference_test(self) -> None:
        result = group_difference_test(make_panel(), "yield_kg_per_ha", "crop")
        assert result is not None and result["p_value"] < 0.001
        same = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "v": [1.0] * 10})
        assert group_difference_test(same, "v", "g") is None

    def test_trends_recover_direction(self) -> None:
        trends = trend_by_group(make_panel(), "yield_kg_per_ha", "year", "crop").set_index("group")
        assert trends.loc["Maize (corn)", "pct_per_period"] > 0
        assert trends.loc["Cassava, fresh", "pct_per_period"] < 0
        assert trends.loc["Maize (corn)", "p_value"] < 0.05

    def test_benchmark_uses_recent_window_and_place_means(self) -> None:
        df = make_panel()
        # Many extra Togo rows must not pull the peer figure: each place counts once.
        df = pd.concat([df] + [df[df["country"] == "Togo"]] * 5, ignore_index=True)
        bench = benchmark_against_peers(df, "yield_kg_per_ha", "crop", "country", "Nigeria", "year", recent_periods=3)
        maize = bench.set_index("group").loc["Maize (corn)"]
        recent = make_panel().query("year >= 2017 and crop == 'Maize (corn)'")
        peer_means = recent[recent["country"] != "Nigeria"].groupby("country")["yield_kg_per_ha"].mean()
        assert maize["window"] == "2017-2019"
        assert maize["peer_median"] == pytest.approx(peer_means.median())
        assert maize["gap_pct"] < 0 and maize["rank"] == 4

    def test_benchmark_needs_enough_peers(self) -> None:
        df = make_panel().query("country in ['Nigeria', 'Ghana']")
        assert benchmark_against_peers(df, "yield_kg_per_ha", "crop", "country", "Nigeria").empty


class TestSummary:
    def test_summary_sentences_match_the_data(self) -> None:
        df = make_panel()
        findings = write_summary(df, "yield_kg_per_ha", "crop", "year", "country", "Nigeria", quality_report(df))
        by_kicker = {f["kicker"]: f for f in findings}
        assert "80 records" in by_kicker["Coverage"]["body"]
        assert "4 countries" in by_kicker["Coverage"]["body"]
        assert "2010-2019" in by_kicker["Coverage"]["body"]
        assert by_kicker["Comparison"]["title"].startswith("Cassava, fresh")
        assert "1 of 2 crops" in by_kicker["Trend"]["body"] and "1 a clear fall" in by_kicker["Trend"]["body"]
        assert "below the peer median on 2 of 2 crops" in by_kicker["Nigeria vs peers"]["title"]
        assert "Data quality" in by_kicker

    def test_summary_omits_what_it_cannot_support(self) -> None:
        df = pd.DataFrame({"yield_tons": np.linspace(1, 3, 12)})
        findings = write_summary(df, "yield_tons")
        assert [f["kicker"] for f in findings] == ["Coverage"]
        assert "2.00 t" in findings[0]["body"]

    def test_non_numeric_measure(self) -> None:
        findings = write_summary(pd.DataFrame({"crop": ["a", "b"]}), "crop")
        assert findings[0]["title"] == "No usable values"
