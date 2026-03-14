"""Utilities for logging litserve request metrics."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_LOGGER_NAME = "litserve.request.metrics"
_LOG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs"))
_LOG_FILE = os.path.join(_LOG_DIR, "request_metrics.log")


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    os.makedirs(_LOG_DIR, exist_ok=True)
    handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_request_metrics(
    module: str,
    duration_ms: float,
    status: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a single request metric entry as JSON."""
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": module,
        "duration_ms": round(duration_ms, 3),
        "status": status,
    }
    if metadata:
        payload.update(metadata)

    _get_logger().info(json.dumps(payload, ensure_ascii=True))
