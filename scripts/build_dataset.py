"""Build the AgroSynapse crop-yield panel from public FAOSTAT and World Bank data.

The committed demo dataset (``data/raw/africa_crop_yield.csv``) is produced by this
script, so every row in the repository can be traced back to a public source:

- **FAOSTAT** *Production: Crops and livestock products* (Africa bulk export) supplies
  crop yield (kg/ha), area harvested (ha) and production (t) per country / crop / year.
- **World Bank** *World Development Indicators* supply the country-year context columns
  (fertilizer use, land use, rural population, agricultural employment, GDP per capita,
  rural electrification, long-run average precipitation).

Usage:
    python scripts/build_dataset.py --output data/raw/africa_crop_yield.csv
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

FAO_URL = (
    "https://bulks-faostat.fao.org/production/"
    "Production_Crops_Livestock_E_Africa.zip"
)
FAO_MEMBER = "Production_Crops_Livestock_E_Africa_NOFLAG.csv"

# FAOSTAT element codes. 5510 (production) is collected for analysis only - it is a
# deterministic function of the target (yield x area) and is blocked from the model
# feature set by scripts/feature_config.py.
FAO_ELEMENTS = {5312: "area_harvested_ha", 5412: "yield_kg_per_ha", 5510: "production_tonnes"}

# Staple and cash crops that dominate African cropland. Restricting to these keeps the
# demo dataset focused and small enough to commit.
CROPS = [
    "Maize (corn)",
    "Sorghum",
    "Millet",
    "Rice",
    "Wheat",
    "Barley",
    "Cassava, fresh",
    "Yams",
    "Sweet potatoes",
    "Potatoes",
    "Cow peas, dry",
    "Beans, dry",
    "Groundnuts, excluding shelled",
    "Soya beans",
    "Sesame seed",
    "Sunflower seed",
    "Seed cotton, unginned",
    "Cocoa beans",
    "Coffee, green",
    "Bananas",
    "Plantains and cooking bananas",
    "Tomatoes",
    "Onions and shallots, dry (excluding dehydrated)",
]

# World Bank WDI indicator codes -> column names used in the panel.
WB_INDICATORS = {
    "fertilizer_kg_per_ha": "AG.CON.FERT.ZS",
    "arable_land_pct": "AG.LND.ARBL.ZS",
    "agri_land_pct": "AG.LND.AGRI.ZS",
    "rural_population_pct": "SP.RUR.TOTL.ZS",
    "agri_employment_pct": "SL.AGR.EMPL.ZS",
    "gdp_per_capita_usd": "NY.GDP.PCAP.KD",
    "rural_electricity_pct": "EG.ELC.ACCS.RU.ZS",
    "avg_precipitation_mm": "AG.LND.PRCP.MM",
}

# FAOSTAT and the World Bank spell a handful of countries differently.
COUNTRY_ALIASES = {
    "united republic of tanzania": "tanzania",
    "democratic republic of the congo": "congo, dem. rep.",
    "congo": "congo, rep.",
    "egypt": "egypt, arab rep.",
    "gambia": "gambia, the",
    "somalia": "somalia, fed. rep.",
}

COLUMN_ORDER = [
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
    "yield_prev_year_kg_per_ha",
    "yield_prev2_year_kg_per_ha",
    "yield_prev3_mean_kg_per_ha",
    "area_prev_year_ha",
    "yield_kg_per_ha",
]

# History features, in years back from the row being predicted.
LAG_SPECS = {
    "yield_prev_year_kg_per_ha": ("yield_kg_per_ha", 1),
    "yield_prev2_year_kg_per_ha": ("yield_kg_per_ha", 2),
    "area_prev_year_ha": ("area_harvested_ha", 1),
}
ROLLING_MEAN_YEARS = 3
PANEL_KEYS = ["country", "crop"]

USER_AGENT = "AgroSynapse/1.0 (dataset builder)"


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_country(name: str) -> str:
    """Fold accents and whitespace so FAOSTAT and World Bank names can be joined."""
    # Punctuation must be folded to ASCII first: the encode("ascii", "ignore") below
    # deletes a typographic apostrophe outright, turning "Cote d’Ivoire" into
    # "cote divoire" and breaking the join against "Cote d'Ivoire".
    unified = str(name).replace("’", "'").replace("ʼ", "'").replace("`", "'")
    folded = unicodedata.normalize("NFKD", unified).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", folded).strip().lower()


def fetch(url: str, timeout: int = 180) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_faostat(start_year: int, end_year: int) -> pd.DataFrame:
    """Download the FAOSTAT Africa bulk export and reshape it into a long panel."""
    LOGGER.info("Downloading FAOSTAT bulk export: %s", FAO_URL)
    archive = zipfile.ZipFile(io.BytesIO(fetch(FAO_URL)))
    with archive.open(FAO_MEMBER) as handle:
        # The FAOSTAT bulk CSVs are UTF-8; reading them as latin-1 silently mangles
        # accented country names (Cote d'Ivoire) and breaks the World Bank join.
        raw = pd.read_csv(handle, encoding="utf-8")

    raw = raw[raw.Item.isin(CROPS) & raw["Element Code"].isin(FAO_ELEMENTS)]
    year_columns = [c for c in raw.columns if re.fullmatch(r"Y\d{4}", c)]

    long = raw.melt(
        id_vars=["Area", "Item", "Element Code"],
        value_vars=year_columns,
        var_name="year",
        value_name="value",
    )
    long["year"] = long["year"].str[1:].astype(int)
    long = long[(long.year >= start_year) & (long.year <= end_year)]
    long = long.dropna(subset=["value"])
    long["metric"] = long["Element Code"].map(FAO_ELEMENTS)

    panel = (
        long.pivot_table(index=["Area", "Item", "year"], columns="metric", values="value")
        .reset_index()
        .rename(columns={"Area": "country", "Item": "crop"})
    )
    panel = panel.dropna(subset=["yield_kg_per_ha"])
    panel = panel[panel.yield_kg_per_ha > 0]
    LOGGER.info(
        "FAOSTAT: %s rows, %s countries, %s crops",
        len(panel),
        panel.country.nunique(),
        panel.crop.nunique(),
    )
    return panel


def load_world_bank(start_year: int, end_year: int) -> pd.DataFrame:
    """Download the World Bank indicators and pivot them to one row per country-year."""
    records: list[tuple[str, str, int, str, float]] = []
    for column, code in WB_INDICATORS.items():
        page = 1
        while True:
            url = (
                f"https://api.worldbank.org/v2/country/all/indicator/{code}"
                f"?format=json&per_page=20000&page={page}"
                f"&date={start_year}:{end_year}"
            )
            payload = json.loads(fetch(url, timeout=120))
            meta, rows = payload[0], payload[1] or []
            for row in rows:
                if row["value"] is None:
                    continue
                records.append(
                    (
                        row["countryiso3code"],
                        row["country"]["value"],
                        int(row["date"]),
                        column,
                        float(row["value"]),
                    )
                )
            if page >= meta["pages"]:
                break
            page += 1
        LOGGER.info("World Bank: fetched %s (%s)", column, code)

    long = pd.DataFrame(records, columns=["iso3", "country", "year", "indicator", "value"])
    wide = long.pivot_table(
        index=["iso3", "country", "year"], columns="indicator", values="value"
    ).reset_index()
    return wide


def build_panel(start_year: int, end_year: int) -> pd.DataFrame:
    fao = load_faostat(start_year, end_year)
    world_bank = load_world_bank(start_year, end_year)

    fao["_key"] = fao.country.map(normalize_country).replace(COUNTRY_ALIASES)
    world_bank["_key"] = world_bank.country.map(normalize_country)

    merged = fao.merge(
        world_bank.drop(columns=["country"]), on=["_key", "year"], how="inner"
    )
    dropped = sorted(set(fao.country) - set(merged.country))
    if dropped:
        # Historical entities (Ethiopia PDR, Sudan (former)) and territories without
        # World Bank coverage legitimately fall out here.
        LOGGER.info("No World Bank match, dropped: %s", ", ".join(dropped))

    merged = merged.drop(columns=["_key"]).rename(columns={"iso3": "country_iso3"})
    merged = add_history_features(merged)
    merged = merged[COLUMN_ORDER].sort_values(["country", "crop", "year"])
    for column in merged.select_dtypes("float").columns:
        merged[column] = merged[column].round(3)
    return merged.reset_index(drop=True)


def add_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach previous-season values for each country/crop series.

    Last season's yield is the single strongest predictor of this season's, and it is
    information an extension officer genuinely has when forecasting. Without it the model
    is asked to rediscover each country/crop's yield level from context variables alone,
    and loses to simply carrying the previous value forward.

    Lags are resolved by an explicit ``year - k`` lookup rather than ``groupby.shift``, so
    a gap in a country's reporting produces a missing value instead of quietly pulling in
    a figure from several seasons earlier.
    """
    frame = panel.copy()
    keyed = frame.set_index(PANEL_KEYS + ["year"])

    for column, (source, lag) in LAG_SPECS.items():
        lookup = keyed[source]
        wanted = pd.MultiIndex.from_arrays(
            [frame[key] for key in PANEL_KEYS] + [frame["year"] - lag]
        )
        frame[column] = lookup.reindex(wanted).to_numpy()

    # Mean of the previous ROLLING_MEAN_YEARS seasons, each looked up the same way, so the
    # window never includes the season being predicted.
    history = [
        keyed["yield_kg_per_ha"]
        .reindex(
            pd.MultiIndex.from_arrays(
                [frame[key] for key in PANEL_KEYS] + [frame["year"] - offset]
            )
        )
        .to_numpy()
        for offset in range(1, ROLLING_MEAN_YEARS + 1)
    ]
    # pandas' mean skips NaN and yields NaN for an all-missing window, where np.nanmean
    # would emit a RuntimeWarning for every series with no history at all.
    frame["yield_prev3_mean_kg_per_ha"] = pd.DataFrame(history).mean(axis=0, skipna=True).to_numpy()

    coverage = frame["yield_prev_year_kg_per_ha"].notna().mean()
    LOGGER.info("Previous-season yield available for %.1f%% of rows", 100 * coverage)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/africa_crop_yield.csv"),
        help="Destination CSV path.",
    )
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    panel = build_panel(args.start_year, args.end_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    LOGGER.info(
        "Wrote %s rows x %s columns to %s (%.1f MB)",
        len(panel),
        panel.shape[1],
        args.output,
        args.output.stat().st_size / 1_048_576,
    )


if __name__ == "__main__":
    main()
