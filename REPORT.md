---
title: "Unit 5 End of Unit Project: Predictive Modeling and Deployment"
subtitle: "An MLOps lifecycle for a risk classification model"
---

# 1. Project objectives and significance

The goal was to build, version, deploy and monitor a classification model the way a production MLOps team would, not just to train a model in a notebook. In healthcare analytics (for example, predicting 30-day readmission risk), a model that is accurate on day one but unversioned, untested or unmonitored can quietly become unsafe as patient populations change. This project demonstrates the full lifecycle: data preparation, modelling, version control, automated CI/CD, a secured REST API, and drift monitoring.

**Dataset note.** The supplied dataset is the Titanic passenger data (891 rows, 12 columns), so the model predicts survival (`Survived`) as a stand-in for a risk outcome. The pipeline is dataset-agnostic: feature lists live in one configuration file.

# 2. Pipeline architecture and tool integration

| Stage | Tool | Output |
|---|---|---|
| Data versioning | DVC | `titanic.csv.dvc`, `dvc.yaml` |
| Preprocessing and features | pandas, scikit-learn | `train.csv`, `test.csv` |
| Training and tracking | scikit-learn, MLflow | `model.joblib`, logged params, metrics, model artifact |
| Automation | `pipeline.py`, GitPython, GitHub Actions | tests, quality gate, Git tag, uploaded artifacts |
| Serving | FastAPI, Uvicorn, Docker | `/predict`, `/health` |
| Monitoring | Evidently AI | HTML drift and performance reports, JSON summary, alert |

Flow: raw data (DVC) -> preprocess -> train (MLflow) -> pytest -> quality gate (ROC-AUC >= 0.80) -> monitoring report -> Git tag -> deploy API -> log predictions -> Evidently drift check -> retrain if alerted.

# 3. Data exploration and preprocessing

* About 38% of passengers survived (class balance 62% / 38%), so accuracy alone is misleading and recall and ROC-AUC matter.
* Strong patterns: survival was 74% for women vs 19% for men, and 63% / 47% / 24% for passenger classes 1 / 2 / 3.
* Missing values: Age 177 (filled by median age per title), Embarked 2 (mode), Cabin 687 (dropped, 77% missing).
* Outliers: Fare clipped at the 1st and 99th percentiles.
* Feature engineering: `Title` extracted from Name, `FamilySize`, `IsAlone`.

# 4. Model development and evaluation

A Random Forest (300 trees, max depth 6, min leaf size 3) in a scikit-learn Pipeline with scaling and one-hot encoding, evaluated on a stratified 20% hold-out set (179 rows).

| Metric | Result |
|---|---|
| Accuracy | 0.8156 |
| Precision | 0.8103 |
| Recall | 0.6812 |
| F1-score | 0.7402 |
| ROC-AUC | 0.8560 |

Recall (0.68) is lower than precision (0.81): the model misses about a third of true positives. If the real use case were readmission, where missing a high-risk patient is costly, the decision threshold should be lowered to trade precision for recall.

# 5. Versioning

Every training run is logged to MLflow (parameters, five metrics, feature list, row count, and the model). DVC tracks the raw data and defines a reproducible two-stage pipeline (`dvc repro`). Git versions all code, and each approved model receives a Git tag containing its timestamp and ROC-AUC.

# 6. CI/CD

`pipeline.py` automates preprocessing, training, tests, a quality gate, monitoring and tagging, and its `--watch` mode re-runs the pipeline when data or code changes. The GitHub Actions workflow triggers on code or data pushes, pull requests, a weekly schedule and manual dispatch, then uploads the model and reports and builds a Docker image. The test suite has 7 automated tests (preprocessing, model quality, API authentication, validation and prediction); all pass.

# 7. Deployment

The FastAPI service exposes `POST /predict` and `GET /health`. Security and robustness features: API-key header authentication (401 without it), Pydantic validation (422 for invalid values such as negative age), the same feature engineering as training, and logging of each request for monitoring. Example response for a 30-year-old first-class woman: `{"prediction": 1, "probability": 0.9797, "risk_level": "high"}`.

# 8. Monitoring

Evidently AI compares training data (reference) with live API traffic. On 60 logged requests, 0 of 10 columns drifted (no alert). A deliberately simulated shift (Age +18 years, Fare x2.5) made 2 of 10 columns (20%) drift, which crossed the 20% threshold and triggered an alert (exit code 1). Reports produced: `data_drift.html`, `data_drift_simulated.html`, `model_performance.html`, `summary.json`.

*[Insert screenshots here: FastAPI /docs page, a successful /predict call, the MLflow run comparison page, the Evidently drift report, and the passing pytest output.]*

# 9. Lessons learned

* Putting preprocessing inside the saved model pipeline prevents training/serving skew.
* MLflow 3 saves models with a "trusted types" check by default; Random Forests required explicitly choosing cloudpickle for models we trained ourselves.
* A drift threshold must be tuned: at 30% the simulated shift went undetected, at 20% it was caught. Too low causes alert fatigue.
* Drift detection works without labels, but performance monitoring needs true outcomes, which can arrive late.
* The small test set (179 rows) makes metrics noisy, so cross-validation would give more reliable estimates.

# 10. Future recommendations

1. Use a real readmission dataset, with fairness checks across demographic groups and privacy controls (HIPAA/GDPR).
2. Tune hyperparameters and compare models (Logistic Regression, XGBoost) with cross-validation, and calibrate probabilities.
3. Use an MLflow tracking server and model registry with staged promotion (Staging, Production).
4. Configure a DVC remote (cloud storage) and schedule monitoring with automated Slack or email alerts.
5. Replace the demo API key with a secrets manager and OAuth, add rate limiting, and store prediction logs in a database.
