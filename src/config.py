"""Central configuration so every script uses the same paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "titanic.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

REPORT_DIR = ROOT / "monitoring" / "reports"
MLFLOW_DB = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "titanic-risk-model"

TARGET = "Survived"
RANDOM_STATE = 42

# Features the model uses (after feature engineering)
NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "Pclass"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# A pipeline run is considered "good enough to deploy" only above this ROC-AUC.
MIN_ROC_AUC_TO_DEPLOY = 0.80
