"""FastAPI inference service for AgroSynapse."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load as joblib_load
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent

# The saved pipeline pickles custom transformers defined in scripts/transform_utils.py.
# joblib resolves them by bare module name, so scripts/ must be importable.
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "model_pipeline.joblib"

app = FastAPI(
    title="AgroSynapse Inference API",
    description="Inference endpoint for agricultural survey predictions.",
    version="0.1.0",
)

pipeline: Optional[Any] = None


class PredictionRow(BaseModel):
    data: Dict[str, Any] = Field(
        ..., description="Field-value mapping for one survey record."
    )


class PredictRequest(BaseModel):
    rows: List[PredictionRow] = Field(
        ..., description="List of survey records to predict."
    )
    target: Optional[str] = Field(
        None,
        description="Optional target column name to ignore when preparing prediction features.",
    )


class PredictResponse(BaseModel):
    predictions: List[float]
    model_path: str
    row_count: int


def load_pipeline(path: Path = MODEL_PATH) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        return joblib_load(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model pipeline from {path}: {exc}") from exc


@app.on_event("startup")
async def startup_event() -> None:
    global pipeline
    pipeline = load_pipeline()


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    if not request.rows:
        raise HTTPException(status_code=400, detail="Request must contain at least one row.")

    records = [row.data for row in request.rows]
    try:
        input_df = pd.DataFrame(records)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {exc}") from exc

    if request.target and request.target in input_df.columns:
        input_df = input_df.drop(columns=[request.target])

    expected_features: List[str] = []
    if hasattr(pipeline, "feature_names_in_"):
        expected_features = [str(x) for x in list(pipeline.feature_names_in_)]

    if expected_features:
        missing = [c for c in expected_features if c not in input_df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Input schema mismatch. Missing required feature columns: "
                    + ", ".join(missing)
                ),
            )
        input_df = input_df[expected_features].copy()

    try:
        predictions = pipeline.predict(input_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        predictions=[float(p) for p in predictions],
        model_path=str(MODEL_PATH),
        row_count=len(request.rows),
    )
