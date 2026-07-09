"""
Inference wrappers for the two registry-managed models.

- badscan:  binary classification (bad_scan / no_bad_scan)
- pagetype: page classification (blank_page / handwriting / photograph / typed_text)

Models are addressed by registry version: callers may pass an explicit
model id (e.g. the version chosen when launching a pass) or fall back to
the role's active version. Loaded weights are validated against the
expected class names; a seed/base model that hasn't been trained on the
task yet yields (None, 0.0) — "unknown" — instead of garbage predictions.
"""

import threading
from pathlib import Path

import numpy as np

from . import db
from .config import BAD_SCAN_CLASSES, all_page_types

_lock = threading.Lock()
_cache: dict[int, tuple[object | None, bool]] = {}  # model_id -> (YOLO, valid)


def _classes_valid(role: str, names: set[str]) -> bool:
    """
    badscan: must be exactly the binary classes.
    pagetype: every class must be a known page type (built-in or custom);
    the model may know only a subset — the LLM covers the rest.
    """
    if role == "badscan":
        return names == set(BAD_SCAN_CLASSES)
    known = set(all_page_types(db.get_settings()))
    return len(names) >= 2 and names <= known


def _load_record(record: dict) -> tuple[object | None, bool]:
    """Return (yolo_model, valid_for_task) for a models-table record."""
    with _lock:
        cached = _cache.get(record["id"])
        if cached is not None:
            return cached

        from ultralytics import YOLO

        path = Path(record["path"])
        if not path.exists():
            entry = (None, False)
        else:
            model = YOLO(str(path))
            names = set(model.names.values()) if isinstance(model.names, dict) else set(model.names)
            valid = model.task == "classify" and _classes_valid(record["role"], names)
            entry = (model, valid)
        _cache[record["id"]] = entry
        return entry


def model_classes(role: str, model_id: int | None = None) -> set[str] | None:
    """Class names the (explicit or active) model can predict, or None if unusable."""
    record = _resolve(role, model_id)
    if record is None:
        return None
    model, valid = _load_record(record)
    if model is None or not valid:
        return None
    names = model.names
    return set(names.values()) if isinstance(names, dict) else set(names)


def _resolve(role: str, model_id: int | None) -> dict | None:
    if model_id is not None:
        record = db.get_model(model_id)
        if record is not None and record["role"] == role:
            return record
        return None
    return db.get_active_model(role)


def invalidate_cache() -> None:
    """Drop cached models (call after activating a different version)."""
    with _lock:
        _cache.clear()


def model_ready(role: str, model_id: int | None = None) -> bool:
    """True if the model (explicit id or active) is trained for the task."""
    record = _resolve(role, model_id)
    if record is None:
        return False
    model, valid = _load_record(record)
    return model is not None and valid


def classify(role: str, image: np.ndarray, model_id: int | None = None) -> tuple[str | None, float]:
    """
    Run a classifier for *role* on a BGR image, using the explicit model
    version when given, otherwise the active one.

    Returns (class_name, confidence) or (None, 0.0) if no usable model.
    """
    record = _resolve(role, model_id)
    if record is None:
        return None, 0.0
    model, valid = _load_record(record)
    if model is None or not valid:
        return None, 0.0

    results = model(image, verbose=False)
    if results and results[0].probs is not None:
        probs = results[0].probs
        return results[0].names[probs.top1], float(probs.top1conf)
    return None, 0.0


def classify_badscan(image: np.ndarray, model_id: int | None = None) -> tuple[str | None, float]:
    return classify("badscan", image, model_id)


def classify_page(image: np.ndarray, model_id: int | None = None) -> tuple[str | None, float]:
    return classify("pagetype", image, model_id)
