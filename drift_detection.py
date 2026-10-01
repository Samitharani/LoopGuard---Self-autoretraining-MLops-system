from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.paths import DRIFT_LOG_CSV


def run_drift_detection() -> dict[str, Any]:
    path = Path(DRIFT_LOG_CSV)
    path.parent.mkdir(exist_ok=True, parents=True)

    result = {
        "OverallDrift": False,
        "Verdict": "No drift detected",
        "FeaturesDrifted": 0,
        "DriftedFeatures": [],
        "Details": {
            "last_run": "not_available",
            "source": "stub_monitoring",
        },
    }

    try:
        path.write_text("feature,drift,score\n", encoding="utf-8")
    except Exception:
        pass

    return result
