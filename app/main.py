"""The PM2.5 alert service.

    uvicorn app.main:app --port 8000

Loads the model once at startup. GET /health says whether this process can
score a reading. POST /predict says whether tomorrow is likely to go above
35 micrograms per cubic metre, and so whether to issue a health alert.
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# The folder holding model.joblib and meta.json.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "model"))


class Reading(BaseModel):
    pm25_today: float = Field(ge=0, le=1000)
    wind_speed_ms: float = Field(ge=0, le=60)
    humidity_pct: float = Field(ge=0, le=100)
    temp_c: float = Field(ge=-40, le=55)
    traffic_index: float = Field(ge=0, le=10)


class Forecast(BaseModel):
    exceed_probability: float
    alert: bool
    threshold: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_DIR / "model.joblib")
    app.state.meta = json.loads((MODEL_DIR / "meta.json").read_text())
    yield


app = FastAPI(title="Smogcast", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    meta = app.state.meta
    matches = meta["sklearn_version"] == sklearn.__version__
    return {
        "status": "ok" if matches else "degraded",
        "model_version": meta["model_version"],
        "trained_with_sklearn": meta["sklearn_version"],
        "running_sklearn": sklearn.__version__,
    }


@app.post("/predict", response_model=Forecast)
def predict(reading: Reading):
    meta = app.state.meta
    row = np.array([[getattr(reading, name) for name in meta["features"]]])
    p = float(app.state.model.predict_proba(row)[0, 1])
    return Forecast(
        exceed_probability=round(p, 4),
        alert=p >= meta["threshold"],
        threshold=meta["threshold"],
        model_version=meta["model_version"],
    )
