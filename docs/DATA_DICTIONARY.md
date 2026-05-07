# Data Dictionary – AgroSynapse

## Survey Data Fields

This dictionary describes the fields in the processed agricultural survey data used by AgroSynapse. Data is stored as a flat CSV file with all survey responses in a single table.

| Field | Type | Description | Example |
| --- | --- | --- | --- |
| farmer_id | integer | Unique identifier for each farmer | 10 |
| full_name | text | Farmer's full name | Pasteur James |
| sex | text | Gender (Male/Female) | Male |
| age | integer | Age in years | 35 |
| phone_number | text | Contact phone number | 7033302085 |
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

