"""Automated tests: run with  pytest -q tests"""
import json
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import config
from src.preprocess import clean_and_engineer, extract_title

client = TestClient(app)
# Use the same key the API uses (from the API_KEY secret/env var, else the local default).
HEADERS = {"X-API-Key": os.getenv("API_KEY", "dev-secret-key")}
GOOD_PATIENT = {"name": "Smith, Mrs. Jane", "pclass": 1, "sex": "female",
                "age": 30, "sibsp": 0, "parch": 0, "fare": 80, "embarked": "C"}


# ---------- preprocessing ----------
def test_extract_title():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Heikkinen, Miss. Laina") == "Miss"
    assert extract_title("Weird, Countess. Someone") == "Rare"


def test_no_missing_values_after_cleaning():
    clean = clean_and_engineer(pd.read_csv(config.RAW_DATA))
    assert clean[config.FEATURES].isna().sum().sum() == 0
    assert (clean["FamilySize"] >= 1).all()


# ---------- model ----------
def test_model_meets_quality_bar():
    metrics = json.loads(config.METRICS_PATH.read_text())
    assert metrics["roc_auc"] >= config.MIN_ROC_AUC_TO_DEPLOY
    assert metrics["accuracy"] > 0.75


# ---------- API ----------
def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_requires_api_key():
    assert client.post("/predict", json=GOOD_PATIENT).status_code == 401


def test_predict_returns_valid_result():
    r = client.post("/predict", json=GOOD_PATIENT, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in ("low", "medium", "high")


def test_predict_rejects_bad_input():
    bad = {**GOOD_PATIENT, "age": -5}
    assert client.post("/predict", json=bad, headers=HEADERS).status_code == 422
