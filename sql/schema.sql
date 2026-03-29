-- AgroSynapse relational schema
-- Target: PostgreSQL 13+

CREATE TABLE IF NOT EXISTS farmers (
    farmer_id SERIAL PRIMARY KEY,
    full_name TEXT,
    sex TEXT,
    age INTEGER CHECK (age >= 0),
    phone TEXT,
    community TEXT,
    district TEXT,
    state TEXT,
    farm_size_ha NUMERIC(10, 2) CHECK (farm_size_ha >= 0),
    registration_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS soil_tests (
    soil_test_id SERIAL PRIMARY KEY,
    farmer_id INTEGER NOT NULL REFERENCES farmers(farmer_id) ON DELETE CASCADE,
    sample_date DATE,
    ph NUMERIC(4, 2),
    nitrogen_ppm NUMERIC(10, 2),
    phosphorus_ppm NUMERIC(10, 2),
    potassium_ppm NUMERIC(10, 2),
    organic_matter_pct NUMERIC(5, 2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS yields (
    yield_id SERIAL PRIMARY KEY,
    farmer_id INTEGER NOT NULL REFERENCES farmers(farmer_id) ON DELETE CASCADE,
    season TEXT,
    crop TEXT,
    area_ha NUMERIC(10, 2) CHECK (area_ha >= 0),
    harvest_date DATE,
    yield_tons NUMERIC(12, 3) CHECK (yield_tons >= 0),
    yield_per_ha NUMERIC(12, 3) GENERATED ALWAYS AS (
        CASE WHEN area_ha IS NULL OR area_ha = 0 THEN NULL
        ELSE yield_tons / area_ha
        END
    ) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_soil_tests_farmer_id ON soil_tests(farmer_id);
CREATE INDEX IF NOT EXISTS idx_yields_farmer_id ON yields(farmer_id);
