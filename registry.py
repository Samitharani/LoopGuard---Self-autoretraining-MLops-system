from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.paths import METRICS_PATH, MODEL_PATH


def list_models() -> list[dict[str, Any]]:
    if not MODEL_PATH.exists():
        return []
    return [{
        "model_name": "LoopGuardBaseline",
        "version": "v1",
        "status": "active",
        "created_at": "2026-01-01T00:00:00Z",
        "metrics": json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {},
    }]


def get_active_model() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    return {"model_name": "LoopGuardBaseline", "version": "v1", "status": "active"}


def ensure_active_model() -> dict[str, Any] | None:
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    if not MODEL_PATH.exists():
        MODEL_PATH.write_text("stub-active-model", encoding="utf-8")
    metrics = {"MAE": 0.15, "RMSE": 0.22, "MAPE": 1.5, "Latency": 10.0, "model_version": "v1"}
    METRICS_PATH.write_text(json.dumps(metrics), encoding="utf-8")
    return get_active_model()
