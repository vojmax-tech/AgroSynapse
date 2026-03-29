# Data Dictionary – AgroSynapse

## Farmers
| Field | Type | Description |
| --- | --- | --- |
| farmer_id | integer | Primary key |
| full_name | text | Farmer name |
| sex | text | Gender |
| age | integer | Farmer age |
| phone | text | Contact number |
| community | text | Community or village |
| district | text | District or LGA |
| state | text | State or region |
| farm_size_ha | numeric | Farm size in hectares |
| registration_date | date | Registration date |

## Soil Tests
| Field | Type | Description |
| --- | --- | --- |
| soil_test_id | integer | Primary key |
| farmer_id | integer | Foreign key to farmers |
| sample_date | date | Soil sampling date |
| ph | numeric | Soil acidity level |
| nitrogen_ppm | numeric | Nitrogen content (ppm) |
| phosphorus_ppm | numeric | Phosphorus content (ppm) |
| potassium_ppm | numeric | Potassium content (ppm) |
| organic_matter_pct | numeric | Organic matter percentage |

## Yields
| Field | Type | Description |
| --- | --- | --- |
| yield_id | integer | Primary key |
| farmer_id | integer | Foreign key to farmers |
| season | text | Season identifier |
| crop | text | Crop type |
| area_ha | numeric | Harvested area in hectares |
| harvest_date | date | Harvest date |
| yield_tons | numeric | Total yield (tons) |
| yield_per_ha | numeric | Derived yield per hectare |
