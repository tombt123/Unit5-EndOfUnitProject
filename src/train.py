"""Steps 2 and 3: train a classifier, evaluate it, and log everything to MLflow.

Run:  python -m src.train
"""
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

PARAMS = {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 3, "random_state": config.RANDOM_STATE}


def build_pipeline() -> Pipeline:
    """Encoding + scaling + model in ONE object, so the API applies identical steps."""
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(**PARAMS))])


def evaluate(model: Pipeline, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def main() -> dict:
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    X_train, y_train = train[config.FEATURES], train[config.TARGET]
    X_test, y_test = test[config.FEATURES], test[config.TARGET]

    mlflow.set_tracking_uri(config.MLFLOW_DB)
    # Pin the artifact folder to THIS project's mlruns/ directory, so the pipeline
    # works on any machine (an old database could otherwise point to a folder that doesn't exist).
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(config.EXPERIMENT_NAME, artifact_location=(config.ROOT / "mlruns").as_uri())
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        model = build_pipeline().fit(X_train, y_train)
        metrics = {k: round(float(v), 4) for k, v in evaluate(model, X_test, y_test).items()}

        mlflow.log_params(PARAMS)
        mlflow.log_param("features", ",".join(config.FEATURES))
        mlflow.log_param("train_rows", len(train))
        mlflow.log_metrics(metrics)
        # cloudpickle is fine here because we trained and stored this model ourselves.
        # (MLflow's default "skops" format refuses RandomForest trees as "untrusted".)
        mlflow.sklearn.log_model(
            model, name="model", input_example=X_train.head(3), serialization_format="cloudpickle"
        )

        # Save a local copy for the API, plus metrics for DVC / CI checks.
        config.MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

        print("MLflow run id:", run.info.run_id)
        print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
