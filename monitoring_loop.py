from __future__ import annotations

from typing import Any

_MONITOR_STATE = {"running": False, "index": 0, "last_row_index": None, "last_timestamp": None}


def start_monitoring_loop(interval_seconds: int = 15) -> dict[str, Any]:
    _MONITOR_STATE["running"] = True
    _MONITOR_STATE["index"] += 1
    return {
        "status": "started",
        "running": True,
        "interval_seconds": interval_seconds,
        "state": _MONITOR_STATE,
    }


def stop_monitoring_loop() -> dict[str, Any]:
    _MONITOR_STATE["running"] = False
    return {"status": "stopped", "running": False, "state": _MONITOR_STATE}
