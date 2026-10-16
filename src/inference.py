"""
Inference wrappers for the two registry-managed models.

- badscan:  binary classification (bad_scan / no_bad_scan)
- pagetype: page classification (blank_page / handwriting / photograph / typed_text),
            either by a whole-page classifier or by a region detector whose
            boxes are mapped to one page type (see :func:`_detect_page_type`)

Models are addressed by registry version: callers may pass an explicit
model id (e.g. the version chosen when launching a pass) or fall back to
the role's active version. Loaded weights are validated against the
expected class names; a seed/base model that hasn't been trained on the
task yet yields (None, 0.0) — "unknown" — instead of garbage predictions.
"""

import json
import threading
from pathlib import Path

import numpy as np

from . import db
from .config import (
    BAD_SCAN_CLASSES,
    PAGETYPE_DETECT_CLASS_MAP,
    PAGETYPE_DETECT_MIN_CONF,
    all_page_types,
)

_lock = threading.Lock()
_cache: dict[int, tuple[object | None, bool]] = {}  # model_id -> (YOLO, valid)


def _names(model) -> list[str]:
    n = model.names
    return list(n.values()) if isinstance(n, dict) else list(n)


def _class_map(record: dict) -> dict[str, str]:
    """Detector class -> page type, from the model's metrics or the default."""
    try:
        custom = json.loads(record.get("metrics") or "{}").get("class_map")
    except ValueError:
        custom = None
    return dict(custom) if isinstance(custom, dict) and custom else dict(PAGETYPE_DETECT_CLASS_MAP)


def _detector_page_types(record: dict, model) -> set[str]:
    """
    Page types a detector can produce: the ones its mapped classes reach,
    plus blank_page (no mapped box on the page).
    """
    cmap = _class_map(record)
    out = {cmap[n] for n in _names(model) if n in cmap}
    return out | {"blank_page"} if out else set()


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
            if model.task == "classify":
                valid = _classes_valid(record["role"], set(_names(model)))
            elif model.task == "detect" and record["role"] == "pagetype":
                valid = _classes_valid("pagetype", _detector_page_types(record, model))
            else:
                valid = False
            entry = (model, valid)
        _cache[record["id"]] = entry
        return entry


def is_detector(role: str, model_id: int | None = None) -> bool:
    """True when the (explicit or active) model for *role* is a usable detector."""
    record = _resolve(role, model_id)
    if record is None:
        return False
    model, valid = _load_record(record)
    return bool(model is not None and valid and model.task == "detect")


def model_classes(role: str, model_id: int | None = None) -> set[str] | None:
    """Class names the (explicit or active) model can predict, or None if unusable."""
    record = _resolve(role, model_id)
    if record is None:
        return None
    model, valid = _load_record(record)
    if model is None or not valid:
        return None
    if model.task == "detect":
        return _detector_page_types(record, model)
    return set(_names(model))


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

    if model.task == "detect":
        return _detect_page_type(model, image, _class_map(record))

    results = model(image, verbose=False)
    if results and results[0].probs is not None:
        probs = results[0].probs
        return results[0].names[probs.top1], float(probs.top1conf)
    return None, 0.0


def _detect_page_type(model, image: np.ndarray, cmap: dict[str, str]
                      ) -> tuple[str | None, float]:
    """
    One page type from a region detector's boxes.

    The mapped class whose boxes cover the most of the page wins; its
    confidence is that class's best box. A page with no mapped box at or
    above PAGETYPE_DETECT_MIN_CONF is blank, with confidence 1 minus the
    best mapped box seen below the floor (so a faint near-miss reads as a
    less certain blank).
    """
    results = model(image, conf=0.05, verbose=False)
    if not results or results[0].boxes is None:
        return None, 0.0
    boxes = results[0].boxes
    names = results[0].names
    area: dict[str, float] = {}
    best: dict[str, float] = {}
    faint = 0.0
    for cls, conf, xywhn in zip(boxes.cls.tolist(), boxes.conf.tolist(),
                                boxes.xywhn.tolist()):
        pt = cmap.get(names[int(cls)])
        if pt is None:
            continue
        if conf < PAGETYPE_DETECT_MIN_CONF:
            faint = max(faint, conf)
            continue
        area[pt] = area.get(pt, 0.0) + xywhn[2] * xywhn[3]
        best[pt] = max(best.get(pt, 0.0), conf)
    if not area:
        return "blank_page", round(1.0 - faint, 4)
    winner = max(area, key=area.get)
    return winner, float(best[winner])


def classify_badscan(image: np.ndarray, model_id: int | None = None) -> tuple[str | None, float]:
    return classify("badscan", image, model_id)


def classify_page(image: np.ndarray, model_id: int | None = None) -> tuple[str | None, float]:
    return classify("pagetype", image, model_id)
