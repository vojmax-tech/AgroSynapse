# Data Dictionary - AgroSynapse

This file documents two schemas:

1. **[Survey template](#survey-template)**: the fields of the farm-survey form the cleaning
   pipeline was designed around (a KoboToolbox-style export). No survey data of this shape is
   shipped with the repository; the table describes what an upload would contain.
2. **[Demo dataset](#demo-dataset-africa-crop-yield-panel)**: the national crop-yield panel the
   app runs on out of the box.

## Survey Template

One row per farmer per season. Personal fields (`full_name`, `phone_number`, `farmer_id`, raw
dates) are never used as model features or grouping columns; see
`scripts/feature_config.py`. Examples below are invented.

| Field | Type | Description | Example |
| --- | --- | --- | --- |
| farmer_id | integer | Unique identifier for each farmer | 10 |
| full_name | text | Farmer's full name | Ada Example |
| sex | text | Gender (Male/Female) | Male |
| age | integer | Age in years | 35 |
| phone_number | text | Contact phone number | 0800 000 0000 |
| community_village | text | Village or community name | Lagelu |
| district_lga | text | District or Local Government Area | Ibadan |
| state_region | text | State or region | Oyo |
| farm_size_ha | numeric | Farm size in hectares | 1.0 |
| registration_date | date | Date farmer was registered | 2026-03-26 |
| sample_date | date | Date soil sample was taken | 2026-03-17 |
| soil_ph | numeric | Soil pH level | 6.0 |
| nitrogen_ppm | numeric | Soil nitrogen content in ppm | 30 |
| phosphorus_ppm | numeric | Soil phosphorus content in ppm | 20 |
| potassium_ppm | numeric | Soil potassium content in ppm | 100 |
| organic_matter | numeric | Soil organic matter percentage | 2.0 |
| season_e_g_2024_maize | text | Season identifier | 2025_CASSAVA |
| crop | text | Main crop type | Cassava |
| area_harvested_ha | numeric | Area harvested in hectares | 0.5 |
| harvest_date | date | Date of harvest | 2025-10-14 |
| yield_tons | numeric | Total yield in tons (target variable) | 6.0 |
| soil_type | text | Soil classification | Peaty |
| uses_fertilizer | text | Whether fertilizer is used (Yes/No) | Yes |
| fertilizer_type | text | Type of fertilizer used | NPK |
| fertilizer_type_organic | integer | Binary flag for organic fertilizer | 0 |
| fertilizer_type_urea | integer | Binary flag for urea fertilizer | 0 |
| fertilizer_type_npk | integer | Binary flag for NPK fertilizer | 1 |
| fertilizer_type_compost | integer | Binary flag for compost fertilizer | 0 |
| irrigation_used | text | Whether irrigation is used (Yes/No) | No |
| improved_seed_used | text | Whether improved seeds are used (Yes/No) | Yes |
| extension_visits_last_12_months | integer | Number of extension visits in last 12 months | 4 |
| market_distance_km | numeric | Distance to market in kilometers | 15.0 |


---

## Demo Dataset: Africa Crop-Yield Panel

The repository ships `data/raw/africa_crop_yield.csv` as its runnable demo dataset. It is
built by `scripts/build_dataset.py` from two public sources and is regenerable at any time:

- **FAOSTAT**, *Production: Crops and Livestock Products* (Africa bulk export) — yield,
  area harvested and production per country / crop / year.
- **World Bank**, *World Development Indicators* — country-year agricultural and economic
  context, joined on country name and year.

One row is one **country x crop x year** observation: 25,305 rows covering 54 countries,
23 crops and the years 1990-2024. The prediction target is `yield_kg_per_ha`.

| Field | Type | Description | Example |
| --- | --- | --- | --- |
| country | text | FAOSTAT reporting country | Nigeria |
| country_iso3 | text | ISO-3166 alpha-3 code (identifier; excluded from features) | NGA |
| crop | text | FAOSTAT crop item | Maize (corn) |
| year | integer | Calendar year of the harvest | 2019 |
| area_harvested_ha | numeric | Area harvested, hectares | 6 200 000 |
| production_tonnes | numeric | Total production, tonnes (**target-derived; excluded from features**) | 11 000 000 |
| fertilizer_kg_per_ha | numeric | Fertilizer consumption per hectare of arable land | 16.4 |
| arable_land_pct | numeric | Arable land as % of land area | 37.3 |
| agri_land_pct | numeric | Agricultural land as % of land area | 77.6 |
| rural_population_pct | numeric | Rural population as % of total | 48.0 |
| agri_employment_pct | numeric | Employment in agriculture, % of total (modelled ILO) | 35.0 |
| gdp_per_capita_usd | numeric | GDP per capita, constant US$ | 2 230 |
| rural_electricity_pct | numeric | Rural population with electricity access, % | 26.0 |
| avg_precipitation_mm | numeric | Long-run average annual precipitation, mm | 1 150 |
| yield_prev_year_kg_per_ha | numeric | Same country/crop yield one year earlier | 1 780 |
| yield_prev2_year_kg_per_ha | numeric | Same country/crop yield two years earlier | 1 755 |
| yield_prev3_mean_kg_per_ha | numeric | Mean yield over the previous three years | 1 762 |
| area_prev_year_ha | numeric | Same country/crop area harvested one year earlier | 6 100 000 |
| **yield_kg_per_ha** | numeric | **Prediction target** — yield in kg per hectare | 1 800 |

### Notes on excluded columns

`production_tonnes` equals `yield_kg_per_ha x area_harvested_ha / 1000`. Supplying it as a
feature lets a model recover the target arithmetically, so `scripts/feature_config.py`
blocks it as target-derived while keeping it available for reporting. `country_iso3` is
blocked as an identifier — it duplicates `country` with no extra information.

The `yield_prev*` columns are **not** leakage: they are observations from earlier seasons,
which a forecaster genuinely has in hand. They are resolved by explicit `year - k` lookup,
so a gap in a country's reporting produces a missing value rather than silently importing a
figure from several seasons back.

### Licensing

FAOSTAT data is published under CC BY 4.0; World Bank WDI under CC BY 4.0. Both require
attribution, which `scripts/build_dataset.py` records in its module docstring.
