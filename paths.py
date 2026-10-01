from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
FRONTEND_DIR = BASE_DIR / "frontend"

DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
FRONTEND_DIR.mkdir(exist_ok=True, parents=True)

MASTER_DATASET_CSV = DATA_DIR / "master_dataset.csv"
PREDICTION_HISTORY_CSV = DATA_DIR / "prediction_history.csv"
DRIFT_LOG_CSV = DATA_DIR / "drift_log.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "active_model.joblib"
