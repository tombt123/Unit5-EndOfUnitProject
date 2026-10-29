"""Step 6: monitoring with Evidently AI.

Compares a "current" dataset against the training "reference" data and writes
interactive HTML reports plus a JSON summary. Exits with code 1 if drift is large,
which is how CI/CD (or a scheduler) gets an alert.

Run:  python -m monitoring.monitor                 # uses logged API predictions if present
      python -m monitoring.monitor --simulate-drift  # demo: shifts the test data on purpose
"""
import argparse
import json
import sys

import pandas as pd
from evidently import DataDefinition, Dataset, BinaryClassification, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

from src import config

DRIFT_ALERT_SHARE = 0.20  # alert if >= 20% of columns drifted
LOG_PATH = config.ROOT / "monitoring" / "prediction_log.csv"


def make_dataset(df: pd.DataFrame, with_labels: bool) -> Dataset:
    definition = DataDefinition(
        numerical_columns=config.NUMERIC_FEATURES,
        categorical_columns=config.CATEGORICAL_FEATURES,
        classification=[BinaryClassification(target=config.TARGET, prediction_labels="prediction",
                                             prediction_probas=None, pos_label=1)] if with_labels else None,
    )
    return Dataset.from_pandas(df, data_definition=definition)


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Demo only: pretend the patient population got older and paid higher fares."""
    drifted = df.copy()
    drifted["Age"] = drifted["Age"] + 18
    drifted["Fare"] = drifted["Fare"] * 2.5
    return drifted


def main(simulate: bool) -> int:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)

    if simulate:
        current = simulate_drift(test)
        source = "simulated drift on test set"
    elif LOG_PATH.exists() and len(pd.read_csv(LOG_PATH)) >= 30:
        current = pd.read_csv(LOG_PATH)
        source = "logged API predictions"
    else:
        current = test
        source = "held-out test set (no or too few API logs yet)"
    print("Current data source:", source)

    cols = config.FEATURES
    # ---- 1. Data drift report ----
    drift_report = Report([DataDriftPreset()])
    snap = drift_report.run(
        Dataset.from_pandas(current[cols], data_definition=DataDefinition(
            numerical_columns=config.NUMERIC_FEATURES, categorical_columns=config.CATEGORICAL_FEATURES)),
        Dataset.from_pandas(reference[cols], data_definition=DataDefinition(
            numerical_columns=config.NUMERIC_FEATURES, categorical_columns=config.CATEGORICAL_FEATURES)),
    )
    name = "data_drift_simulated" if simulate else "data_drift"
    snap.save_html(str(config.REPORT_DIR / f"{name}.html"))
    result = snap.dict()

    # Find the "share of drifted columns" metric in the result
    share = None
    for m in result["metrics"]:
        if m["metric_name"].startswith("DriftedColumnsCount"):
            share = m["value"]["share"]
            count = m["value"]["count"]
    print(f"Drifted columns: {int(count)} of {len(cols)} ({share:.0%})")

    # ---- 2. Model performance report (needs labels) ----
    if "Survived" in current.columns and not simulate:
        import joblib
        model = joblib.load(config.MODEL_PATH)
        cur = current.copy()
        cur["prediction"] = model.predict(cur[cols])
        ref = reference.copy()
        ref["prediction"] = model.predict(ref[cols])
        perf = Report([ClassificationPreset()])
        psnap = perf.run(make_dataset(cur[cols + ["Survived", "prediction"]], True),
                         make_dataset(ref[cols + ["Survived", "prediction"]], True))
        psnap.save_html(str(config.REPORT_DIR / "model_performance.html"))

    alert = share is not None and share >= DRIFT_ALERT_SHARE
    summary = {"data_source": source, "drifted_share": share, "drifted_columns": int(count),
               "alert": bool(alert), "threshold": DRIFT_ALERT_SHARE}
    (config.REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("ALERT: significant data drift detected - consider retraining!" if alert else "No significant drift.")
    return 1 if alert else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-drift", action="store_true")
    sys.exit(main(parser.parse_args().simulate_drift))
