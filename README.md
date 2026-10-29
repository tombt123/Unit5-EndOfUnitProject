# MLOps Lifecycle Project: Risk Classification, Versioning, CI/CD, API and Monitoring

An end-to-end machine learning pipeline built with **Python, scikit-learn, MLflow, DVC, FastAPI and Evidently AI**.

> **Note on the dataset.** The scenario describes hospital readmission risk, but the provided data is the
> Titanic passenger dataset. The pipeline therefore predicts `Survived` (1 = survived) and presents the result as a
> generic "risk" classifier. Nothing in the pipeline is Titanic-specific: swap in a real readmission
> dataset by editing `src/config.py` and the feature code in `src/preprocess.py` and `api/main.py`.

## Project layout

```
mlops_project/
├── data/raw/titanic.csv        raw data (tracked by DVC via titanic.csv.dvc)
├── src/
│   ├── download_data.py        downloads titanic.csv from Google Drive if missing
│   ├── config.py               paths, feature lists, settings (one place to change things)
│   ├── preprocess.py           exploration, cleaning, feature engineering, train/test split
│   └── train.py                model training, evaluation, MLflow logging
├── api/main.py                 FastAPI service (/health, /predict) with API-key security
├── monitoring/monitor.py       Evidently drift + performance reports and drift alert
├── pipeline.py                 Python automation: preprocess -> train -> test -> gate -> monitor -> git tag
├── tests/test_pipeline.py      automated tests (preprocessing, model quality, API)
├── dvc.yaml                    DVC pipeline definition
├── .github/workflows/ci.yml    GitHub Actions CI/CD
├── Dockerfile                  container for deployment
└── requirements.txt
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Training steps

The dataset is downloaded automatically from Google Drive by `src/download_data.py` (it skips the download if
`data/raw/titanic.csv` already exists). The Drive file must be shared as "Anyone with the link". If the download is
blocked, place `titanic.csv` in `data/raw/` manually.

```bash
python -m src.preprocess     # explores data, cleans it, writes data/processed/train.csv and test.csv
python -m src.train          # trains Random Forest, prints metrics, logs run to MLflow
```

Or run everything at once: `python pipeline.py`.

**Preprocessing decisions**

| Issue | Treatment |
|---|---|
| Age missing (20%) | Filled with the median age of the passenger's title (Mr, Mrs, Miss, Master, Rare) |
| Embarked missing (2 rows) | Filled with the most common port |
| Cabin missing (77%) | Dropped (too sparse to trust) |
| Fare outliers | Clipped to the 1st and 99th percentile |
| Categorical columns | One-hot encoded inside the model pipeline (`handle_unknown="ignore"`) |
| Engineered features | `Title` (from Name), `FamilySize`, `IsAlone` |

**Model:** Random Forest (300 trees, max depth 6) inside a scikit-learn `Pipeline`, so scaling and encoding are
saved with the model and the API cannot apply different preprocessing than training did.
Evaluated on a stratified 20% hold-out set (random_state=42).

| Metric | Result |
|---|---|
| Accuracy | 0.8156 |
| Precision | 0.8103 |
| Recall | 0.6812 |
| F1 | 0.7402 |
| ROC-AUC | 0.8560 |

## 3. Version control (MLflow + DVC + Git)

* **MLflow** logs parameters, metrics and the model artifact for every run.
  View runs with: `mlflow ui --backend-store-uri sqlite:///mlflow.db` then open http://127.0.0.1:5000
* **DVC** versions the data and the pipeline. `dvc add data/raw/titanic.csv` created `titanic.csv.dvc`
  (a small pointer file stored in Git instead of the data). `dvc repro` re-runs only the stages whose inputs changed.
  Configure remote storage with `dvc remote add -d storage <path-or-url>` then `dvc push`.
* **Git** versions the code. `pipeline.py` creates a Git tag (using GitPython) for each approved model.

## 4. CI/CD pipeline

Two layers, both running the same stages:

1. **Local Python automation** (`pipeline.py`): preprocess, train, run tests, check the **quality gate**
   (ROC-AUC must be at least 0.80, otherwise the pipeline stops), generate the monitoring report, tag the release.
   `python pipeline.py --watch` re-runs everything automatically when the data or any `src/*.py` file changes.
2. **GitHub Actions** (`.github/workflows/ci.yml`): triggers on pushes that touch code or data, on pull requests,
   every Monday (scheduled retraining), and manually. It trains, tests, builds reports, uploads the model as an artifact,
   and builds the Docker image on `main`.

## 5. Deploying the API

```bash
export API_KEY="choose-a-strong-secret"     # Windows PowerShell: $env:API_KEY="..."
uvicorn api.main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"name":"Smith, Mrs. Jane","pclass":1,"sex":"female","age":30,"sibsp":0,"parch":0,"fare":80,"embarked":"C"}'
# {"prediction":1,"probability":0.9797,"risk_level":"high"}
```

**Security and testability:** requests without the correct `X-API-Key` get `401`; Pydantic validates inputs
(a negative age returns `422`); the `/health` endpoint supports uptime checks; `pytest` covers the API using FastAPI's `TestClient`.
Docker: `docker build -t risk-api . && docker run -p 8000:8000 -e API_KEY=secret risk-api`.

## 6. Monitoring strategy (Evidently AI)

Every `/predict` call is appended to `monitoring/prediction_log.csv`. Running

```bash
python -m monitoring.monitor                  # reference (training data) vs. logged API traffic
python -m monitoring.monitor --simulate-drift # demo: deliberately shifted data to show an alert
```

produces interactive HTML reports in `monitoring/reports/`:

* `data_drift.html`: per-feature drift tests (reference vs. current data)
* `data_drift_simulated.html`: the same report with simulated drift
* `model_performance.html`: classification metrics, confusion matrix, ROC for current vs. reference
* `summary.json`: machine-readable result

**Alerting:** if 20% or more of the columns drift, the script prints an alert and exits with code 1, so a scheduler or
the CI job fails visibly. In the demo, shifting `Age` and `Fare` triggers the alert while normal data does not.
In production, schedule the script (cron, GitHub Actions) and route failures to email or Slack, then retrain with `python pipeline.py`.
Note that the performance report needs true outcomes, which in healthcare often arrive days or weeks later.

## 7. Best practices followed

* **Reproducibility:** fixed random seeds, pinned split, one `config.py`, DVC-tracked data, MLflow-tracked runs, `requirements.txt`.
* **No training/serving skew:** encoding and scaling live inside the saved model pipeline; the API reuses `extract_title` from training code.
* **Quality gates:** automated tests plus a minimum-metric check before anything is deployed.
* **Scalability:** the API is stateless and containerised, so it can run as several replicas behind a load balancer.
* **Security:** API-key authentication, strict input validation, secrets via environment variables (never committed).
* **Limitations / next steps:** the demo key and file-based logging are for learning; production should use a secrets manager,
  a database for logs, an MLflow server with a model registry, and rate limiting. For real patient data, add privacy controls (HIPAA/GDPR),
  check fairness across groups, and calibrate the probability thresholds with clinicians.
