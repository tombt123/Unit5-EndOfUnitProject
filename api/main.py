"""Step 5: serve the trained model with FastAPI.

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
Security: every /predict call must send the header  X-API-Key: <your key>
          (set the key with the API_KEY environment variable; default is for local testing only).
"""
import csv
import os
from datetime import datetime, timezone
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src import config
from src.preprocess import extract_title

API_KEY = os.getenv("API_KEY", "dev-secret-key")
LOG_PATH = config.ROOT / "monitoring" / "prediction_log.csv"

app = FastAPI(title="Risk Prediction API", version="1.0.0")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_model = None


def get_model():
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run the pipeline first.")
        _model = joblib.load(config.MODEL_PATH)
    return _model


def check_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class PatientInput(BaseModel):
    """Input validation: bad values are rejected automatically with a 422 error."""
    name: str = Field(..., examples=["Smith, Mr. John"])
    pclass: int = Field(..., ge=1, le=3)
    sex: Literal["male", "female"]
    age: float = Field(..., ge=0, le=120)
    sibsp: int = Field(0, ge=0)
    parch: int = Field(0, ge=0)
    fare: float = Field(..., ge=0)
    embarked: Literal["S", "C", "Q"]


class Prediction(BaseModel):
    prediction: int
    probability: float
    risk_level: Literal["low", "medium", "high"]


def to_features(p: PatientInput) -> pd.DataFrame:
    """Apply the same feature engineering used in training."""
    family = p.sibsp + p.parch + 1
    row = {
        "Age": p.age, "Fare": p.fare, "SibSp": p.sibsp, "Parch": p.parch,
        "FamilySize": family, "IsAlone": int(family == 1), "Pclass": p.pclass,
        "Sex": p.sex, "Embarked": p.embarked, "Title": extract_title(p.name),
    }
    return pd.DataFrame([row])[config.FEATURES]


def log_prediction(features: pd.DataFrame, pred: int, proba: float) -> None:
    """Save every request so Evidently can compare live data with training data."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    new_file = not LOG_PATH.exists()
    row = features.iloc[0].to_dict()
    row.update(prediction=pred, probability=round(proba, 4),
               timestamp=datetime.now(timezone.utc).isoformat())
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction, dependencies=[Depends(check_key)])
def predict(patient: PatientInput):
    model = get_model()
    X = to_features(patient)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    log_prediction(X, pred, proba)
    level = "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low"
    return Prediction(prediction=pred, probability=round(proba, 4), risk_level=level)
