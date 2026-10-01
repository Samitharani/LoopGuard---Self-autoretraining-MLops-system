# LoopGuard Backend

Real-Time ML Model Monitoring and Auto-Retraining System.
Use case: food-delivery demand prediction (Swiggy/Zomato/Uber-Eats style),
used as a vehicle to demonstrate the actual product — an MLOps monitoring
and auto-retraining platform.

## Folder Structure

```
backend/
├── data/                     # Raw datasets (+ generate_sample_data.py)
├── preprocessing/            # Step 2: cleaning scripts per dataset
├── feature_engineering/      # Step 3: time + contextual features
├── dataset/                  # Step 4: merge into master_dataset.csv
├── eda/                      # Step 5: exploratory data analysis
├── models/                   # Step 6: training + saved model artifacts
├── monitoring/               # Steps 7-8: live simulation + drift detection
├── retraining/               # Step 9: auto-retraining
├── registry/                 # Step 10: model version registry
├── api/                       # Step 11: Pydantic schemas for the API
├── utils/                    # Shared logger + path constants
├── outputs/                   # All generated CSVs, charts, metrics
├── logs/                     # Rotating log files per module
├── tests/                    # Unit tests (pytest)
├── app.py                    # Step 11: FastAPI entrypoint
└── requirements.txt
```

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Full Pipeline (first time, in order)

```bash
# 1. Generate dummy datasets (skip this once you have real exported CSVs
#    with the same column names, dropped into data/)
python data/generate_sample_data.py

# 2. Clean all 5 raw datasets
python preprocessing/run_preprocessing.py

# 3. Build time + contextual features
python feature_engineering/feature_engineering.py

# 4. Merge into the hourly-per-district master dataset
python dataset/merge_data.py

# 5. (Optional) Exploratory data analysis + charts
python eda/eda.py

# 6. Train the model
python models/train_model.py

# 6b. Register it as the first Active model version (one-time, first run only)
python models/register_initial_model.py

# 7. Simulate live traffic (Ctrl+C to stop early)
python monitoring/simulate_live_data.py --rows 30 --interval 2

# 8. Check for drift
python monitoring/drift_detection.py

# 9. Retrain if needed (also runs automatically via the /retrain API)
python retraining/retrain_model.py
```

## Run the API

```bash
uvicorn app:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

### Endpoints

| Method | Path        | Purpose                                        |
|--------|-------------|-------------------------------------------------|
| GET    | /dashboard  | Summary for the dashboard landing page          |
| GET    | /live       | Most recent live predictions                    |
| GET    | /drift      | Run drift detection now, return the verdict     |
| POST   | /retrain    | Trigger auto-retraining                         |
| GET    | /models     | List all registered model versions              |
| GET    | /metrics    | Latest metrics of the active model              |
| GET    | /history    | Full prediction history (paginated)             |

All responses are JSON — ready to be consumed by the React frontend
(Control Center, Live Monitoring, Drift Detection, Retraining, Model
Registry, Data Sources, Alerts, Settings pages).

## Testing

```bash
pytest tests/ -v
```

## Swapping in Real Data

Every preprocessing script validates against the exact column names
listed in the project spec. Drop real CSVs into `data/` with those
same column names and re-run the pipeline from Step 2 onward — no
code changes required unless your schema differs.

## Notes on Design Decisions

- **Demand target**: the raw datasets are order-level, but the model
  predicts *demand* (order volume) per District/Hour, so
  `dataset/merge_data.py` aggregates order-level rows into an
  hourly-per-district table before training.
- **Retraining safety**: `retraining/retrain_model.py` only replaces
  the production model if the new model's MAE is equal to or better
  than the currently active model's MAE. Rejected retrains are still
  logged in the registry for auditability, and the previous model is
  restored automatically.
- **Drift detection**: uses both the Kolmogorov-Smirnov test (are the
  distributions statistically different?) and PSI (how large is the
  shift?) per feature, so a single noisy feature doesn't trigger a
  false alarm — retraining is recommended only once a configurable
  fraction of monitored features show drift.
