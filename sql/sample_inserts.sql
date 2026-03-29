-- Sample inserts for AgroSynapse

INSERT INTO farmers (full_name, sex, age, phone, community, district, state, farm_size_ha, registration_date)
VALUES
    ('Amina Bello', 'F', 42, '0800000000', 'Kofa', 'Dutse', 'Jigawa', 2.5, '2024-05-12'),
    ('Chinedu Okafor', 'M', 35, '0811111111', 'Umuahia', 'Umuahia North', 'Abia', 4.1, '2024-05-14');

INSERT INTO soil_tests (farmer_id, sample_date, ph, nitrogen_ppm, phosphorus_ppm, potassium_ppm, organic_matter_pct)
VALUES
    (1, '2024-06-01', 6.2, 18.4, 9.1, 12.7, 2.3),
    (2, '2024-06-03', 5.8, 21.0, 8.4, 11.9, 2.8);

INSERT INTO yields (farmer_id, season, crop, area_ha, harvest_date, yield_tons)
VALUES
    (1, '2024_MAIZE', 'maize', 2.5, '2024-11-10', 6.4),
    (2, '2024_MAIZE', 'maize', 4.1, '2024-11-14', 10.2);
