"""
Feedback-driven retraining ("reinforcement" loop).

This is dataset aggregation + periodic retraining, NOT per-flag online
learning: every reviewer verdict is a permanent labeled example, and a
retrain always rebuilds from the FULL accumulated set. So flags, missed
flags, and corrections all feed the next model version.

Verdict -> label mapping:
  badscan role  — true_positive / false_negative  -> "bad_scan"
                  false_positive / true_negative  -> "no_bad_scan"
  pagetype role — the reviewer-confirmed page type

Additional example sources:
  badscan  — synthetic degraded images under SYNTHETIC_DIR/badscan/<label>/
  pagetype — unreviewed LLM-classified scans (weak bootstrap labels)

Each retrain snapshots its examples into a versioned dataset (browsable
and reproducible in the app), materializes them to an ultralytics
classification layout, fine-tunes from the current active model when it
is already task-trained (so corrections compound), registers the result
as a new model version, and optionally activates it.
"""

import random
import shutil
import threading
import traceback
from collections import Counter
from pathlib import Path

from . import db, inference, registry
from .config import (
    BAD_SCAN_CLASSES,
    BADSCAN_BASE_WEIGHTS,
    DATASETS_DIR,
    PAGETYPE_BASE_WEIGHTS,
    SYNTHETIC_DIR,
    TRAIN_RUNS_DIR,
    all_page_types,
)

MIN_PER_CLASS = 4
VAL_FRACTION = 0.15

_train_lock = threading.Lock()
progress: dict = {}

ROLE_BASE_WEIGHTS = {
    "badscan": BADSCAN_BASE_WEIGHTS,
    "pagetype": PAGETYPE_BASE_WEIGHTS,
}


def is_running() -> bool:
    return _train_lock.locked()


# ---------------------------------------------------------------------------
# Example collection
# ---------------------------------------------------------------------------

def _synthetic_examples(role: str) -> list[dict]:
    """Pre-labeled images kept under SYNTHETIC_DIR/<role>/<label>/*."""
    out = []
    base = SYNTHETIC_DIR / role
    if not base.is_dir():
        return out
    for label_dir in base.iterdir():
        if not label_dir.is_dir():
            continue
        for img in label_dir.iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                out.append({"filepath": str(img), "jpeg_path": str(img),
                            "label": label_dir.name, "source": "synthetic"})
    return out


def _examples(role: str) -> list[dict]:
    if role == "badscan":
        examples = db.labeled_badscan_examples() + _synthetic_examples("badscan")
    else:
        examples = db.labeled_pagetype_examples() + _synthetic_examples("pagetype")
    # An example is only usable if its image still exists on disk.
    return [e for e in examples if e["jpeg_path"] and Path(e["jpeg_path"]).exists()]


def _role_classes(role: str, examples: list[dict]) -> list[str]:
    """
    Classes the retrained model will learn.

    badscan: the fixed binary pair (both must have enough examples).
    pagetype: every label with enough examples — this is how custom
    classes get folded into the model over time.
    """
    counts = Counter(e["label"] for e in examples)
    if role == "badscan":
        return list(BAD_SCAN_CLASSES)
    return sorted(c for c, n in counts.items() if n >= MIN_PER_CLASS)


def dataset_summary(role: str) -> dict:
    """Label counts currently available for training *role*."""
    examples = _examples(role)
    counts = Counter(e["label"] for e in examples)

    if role == "badscan":
        display = {c: counts.get(c, 0) for c in BAD_SCAN_CLASSES}
        trainable = all(counts.get(c, 0) >= MIN_PER_CLASS for c in BAD_SCAN_CLASSES)
    else:
        display = {c: counts.get(c, 0) for c in all_page_types(db.get_settings())}
        for c, n in counts.items():
            display.setdefault(c, n)
        trainable = len(_role_classes(role, examples)) >= 2

    return {
        "counts": display,
        "total": len(examples),
        "trainable": trainable,
        "min_per_class": MIN_PER_CLASS,
        "human": sum(1 for e in examples if e.get("source") == "human"),
        "llm": sum(1 for e in examples if e.get("source") == "llm"),
        "synthetic": sum(1 for e in examples if e.get("source") == "synthetic"),
    }


# ---------------------------------------------------------------------------
# Dataset snapshots (versioned, browsable, reproducible)
# ---------------------------------------------------------------------------

def build_dataset(role: str, notes: str = "") -> dict:
    """
    Snapshot the current labeled examples into a new immutable dataset
    version (train/val split assigned per example) and return its record.
    """
    examples = _examples(role)
    classes = _role_classes(role, examples)
    counts = Counter(e["label"] for e in examples)

    if role == "badscan":
        for cls in classes:
            if counts.get(cls, 0) < MIN_PER_CLASS:
                raise ValueError(
                    f"Not enough examples for '{cls}': {counts.get(cls, 0)}/{MIN_PER_CLASS}. "
                    "Review more scans (or add synthetic bad scans) first."
                )
    elif len(classes) < 2:
        raise ValueError(
            "Need at least two page-type classes with "
            f">= {MIN_PER_CLASS} examples each; currently: {dict(counts)}."
        )

    rng = random.Random(1337)
    by_class: dict[str, list[dict]] = {c: [] for c in classes}
    for e in examples:
        if e["label"] in by_class:
            by_class[e["label"]].append(e)

    items: list[dict] = []
    for cls, group in by_class.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * VAL_FRACTION))
        for i, e in enumerate(group):
            items.append({
                "label": cls,
                "source": e.get("source", ""),
                "image_path": e["jpeg_path"],
                "split": "val" if i < n_val else "train",
            })

    version = db.next_dataset_version(role)
    ds_id = db.create_dataset(role, version, classes, notes or
                              f"Snapshot of {len(items)} labeled examples.", items)
    return db.get_dataset(ds_id)


def materialize_dataset(dataset_id: int) -> Path:
    """Copy a dataset snapshot's items into an ultralytics classify layout."""
    ds = db.get_dataset(dataset_id)
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")
    ds_dir = DATASETS_DIR / ds["version"]
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    for it in db.dataset_items(dataset_id):
        src = Path(it["image_path"])
        if not src.exists():
            continue
        dest = ds_dir / it["split"] / it["label"]
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / f"{it['id']:06d}_{src.name}")
    return ds_dir


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _base_weights(role: str) -> tuple[str, str]:
    """
    Weights to fine-tune from: the current active model when it is already
    trained on this task (continual learning), otherwise the stock base.
    """
    active = db.get_active_model(role)
    if active and inference.model_ready(role) and Path(active["path"]).exists():
        return active["path"], active["version"]
    return ROLE_BASE_WEIGHTS[role], ROLE_BASE_WEIGHTS[role]


def train(role: str, epochs: int = 50, auto_activate: bool = True,
          dataset_id: int | None = None) -> dict | None:
    """
    Blocking retrain for *role*. Builds a fresh dataset snapshot unless a
    *dataset_id* is given. Returns the new model record, or None if a
    training run is already in progress. Raises on dataset/training errors.
    """
    if not _train_lock.acquire(blocking=False):
        return None

    global progress
    run_id = None
    try:
        if dataset_id is None:
            ds = build_dataset(role)
        else:
            ds = db.get_dataset(dataset_id)
            if ds is None or ds["role"] != role:
                raise ValueError("Invalid dataset for this role")
        dataset_id = ds["id"]

        import json
        ds_info = {"version": ds["version"], "classes": json.loads(ds["classes"]),
                   "counts": json.loads(ds["counts"]), "total": ds["total"]}
        run_id = db.create_training_run(role, epochs, ds_info, dataset_id=dataset_id)
        progress = {"role": role, "dataset": ds["version"], "epochs": epochs,
                    "status": "materializing"}

        ds_dir = materialize_dataset(dataset_id)
        version = db.next_model_version(role)
        progress = {"role": role, "version": version, "dataset": ds["version"],
                    "epochs": epochs, "status": "training", "counts": ds_info["counts"]}
        print(f"[retrain] {version} from {ds['version']}: {ds_info['counts']}")

        weights, base_name = _base_weights(role)
        from ultralytics import YOLO

        model = YOLO(weights)
        results = model.train(
            data=str(ds_dir),
            epochs=epochs,
            imgsz=640,
            project=str(TRAIN_RUNS_DIR),
            name=version,
            exist_ok=True,
            verbose=True,
        )

        best = Path(results.save_dir) / "weights" / "best.pt"
        if not best.exists():
            raise RuntimeError(f"Training produced no weights at {best}")

        metrics = {"dataset": ds_info, "epochs": epochs, "dataset_version": ds["version"]}
        try:
            metrics["top1_accuracy"] = float(results.results_dict.get("metrics/accuracy_top1", 0))
        except Exception:
            pass

        record = registry.register_version(
            role, best, base_model=base_name, metrics=metrics,
            notes=f"Trained on {ds['version']} ({ds_info['total']} examples).",
            activate=auto_activate,
        )
        if auto_activate:
            inference.invalidate_cache()

        db.finish_training_run(run_id, "done", model_id=record["id"])
        progress = {}
        print(f"[retrain] registered {record['version']} (active={auto_activate})")
        return record

    except Exception as e:
        traceback.print_exc()
        if run_id is not None:
            db.finish_training_run(run_id, "error", error=str(e))
        progress = {}
        raise
    finally:
        _train_lock.release()


def train_async(role: str, epochs: int, auto_activate: bool,
                dataset_id: int | None = None) -> threading.Thread:
    """Fire-and-forget retrain for the web app."""
    def _run():
        try:
            train(role, epochs=epochs, auto_activate=auto_activate, dataset_id=dataset_id)
        except Exception:
            pass  # already logged + recorded in training_runs

    t = threading.Thread(target=_run, daemon=True, name=f"retrain-{role}")
    t.start()
    return t
