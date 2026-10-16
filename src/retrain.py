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

Examples no longer come from disk layout or from re-deriving reviews on
every build. :mod:`src.examples` owns the ``training_examples`` table, which
is synced from every source (reviewer verdicts, curated imports, synthetic
positives, LLM page types) and is editable in the app — so a corrected or
deactivated label is honoured here without any special-casing.

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

from . import db, examples, inference, registry
from .config import (
    BAD_SCAN_CLASSES,
    BADSCAN_BASE_WEIGHTS,
    DATASETS_DIR,
    PAGETYPE_BASE_WEIGHTS,
    TRAIN_RUNS_DIR,
    all_page_types,
)

MIN_PER_CLASS = 4
VAL_FRACTION = 0.15
# Longest side kept when materializing a dataset. Training runs at imgsz=640,
# so full 2500px archival derivatives are pure cost — they made the snapshot
# ~528MB and timed out the upload to the remote trainer. 1024 keeps the
# snapshot comfortably viewable in the dataset browser.
MATERIALIZE_MAX_SIDE = 1024

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

def _examples(role: str) -> list[dict]:
    """
    Active, on-disk examples for *role*. Syncing first means files or
    verdicts added since the last build are picked up, while corrections
    already recorded in the store are preserved (sync is insert-only).
    """
    examples.sync(role)
    return examples.for_training(role)


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

    by_source: Counter = Counter(e.get("source", "") for e in examples)
    store = db.training_example_stats(role)
    return {
        "counts": display,
        "total": len(examples),
        "trainable": trainable,
        "min_per_class": MIN_PER_CLASS,
        "by_source": dict(by_source),
        # Kept as flat keys for the existing dataset panel.
        "human": by_source.get("human", 0),
        "llm": by_source.get("llm", 0),
        "synthetic": by_source.get("synthetic", 0),
        "curated": by_source.get("curated", 0),
        "corrected": store.get("corrected", 0),
        "excluded": store.get("inactive", 0),
    }


# ---------------------------------------------------------------------------
# Dataset snapshots (versioned, browsable, reproducible)
# ---------------------------------------------------------------------------

def _is_model_error(e: dict) -> bool:
    """
    True when a previous model got this image wrong: the reviewer either
    cleared something it flagged (false_positive) or caught something it
    missed (false_negative). These are the most informative examples we
    have, so they are never sampled out.
    """
    return e.get("verdict") in ("false_positive", "false_negative")


def select_badscan_examples(examples: list[dict], settings: dict | None = None,
                            seed: int | None = None) -> tuple[list[dict], dict]:
    """
    Choose the bad-scan training set under the standing policy:

      * every bad_scan example is always included;
      * every example a previous model got wrong is always included, even
        when it is a clean page (those are the hard negatives);
      * the remaining clean pages are sampled at random, spread evenly
        across projects, to `neg_sample_ratio` times the positive count.

    The sample is deliberately re-drawn on every build (unless
    `neg_sample_seed` is set), so successive dataset versions see different
    clean pages and the model is not fitted to one fixed slice of them.

    Returns (chosen, report).
    """
    settings = settings if settings is not None else db.get_settings()
    try:
        ratio = float(settings.get("neg_sample_ratio") or 1.5)
    except ValueError:
        ratio = 1.5
    if seed is None:
        raw = (settings.get("neg_sample_seed") or "").strip()
        seed = int(raw) if raw.isdigit() else None
    rng = random.Random(seed)  # seed=None -> fresh sample each build

    always, pool = [], []
    for e in examples:
        if e["label"] == "bad_scan" or _is_model_error(e):
            always.append(e)
        else:
            pool.append(e)

    n_pos = sum(1 for e in always if e["label"] == "bad_scan")
    target = max(MIN_PER_CLASS, int(round(n_pos * ratio)))
    # Negatives already forced in (cleared false positives) count toward it.
    forced_neg = len(always) - n_pos
    need = max(0, target - forced_neg)

    # Spread the draw evenly over projects so no single volume dominates.
    by_project: dict[int | None, list[dict]] = {}
    for e in pool:
        by_project.setdefault(e.get("project_id"), []).append(e)
    for group in by_project.values():
        rng.shuffle(group)

    sampled: list[dict] = []
    if need and by_project:
        keys = sorted(by_project, key=lambda k: (k is None, k))
        i = 0
        while len(sampled) < need and any(by_project[k] for k in keys):
            group = by_project[keys[i % len(keys)]]
            if group:
                sampled.append(group.pop())
            i += 1

    chosen = always + sampled
    report = {
        "positives": n_pos,
        "model_errors": sum(1 for e in always if _is_model_error(e)),
        "forced_negatives": forced_neg,
        "sampled_negatives": len(sampled),
        "negative_pool": len(pool),
        "neg_sample_ratio": ratio,
        "neg_sample_seed": seed,
        "per_project_sampled": {
            str(k): sum(1 for e in sampled if e.get("project_id") == k)
            for k in sorted(by_project, key=lambda k: (k is None, k))
        },
    }
    return chosen, report


def build_dataset(role: str, notes: str = "") -> dict:
    """
    Snapshot the current labeled examples into a new immutable dataset
    version (train/val split assigned per example) and return its record.

    For badscan the snapshot is a *selection*, not the whole store — see
    :func:`select_badscan_examples`. Page-type keeps every example.
    """
    examples = _examples(role)
    selection_report: dict = {}
    if role == "badscan":
        examples, selection_report = select_badscan_examples(examples)
    classes = _role_classes(role, examples)
    counts = Counter(e["label"] for e in examples)

    if role == "badscan":
        for cls in classes:
            if counts.get(cls, 0) < MIN_PER_CLASS:
                raise ValueError(
                    f"Not enough examples for '{cls}': {counts.get(cls, 0)}/{MIN_PER_CLASS}. "
                    "Import or review more scans first."
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
    if selection_report:
        detail = (f"{selection_report['positives']} bad, "
                  f"{selection_report['model_errors']} past model errors, "
                  f"{selection_report['sampled_negatives']} clean sampled "
                  f"of {selection_report['negative_pool']} "
                  f"(ratio {selection_report['neg_sample_ratio']})")
        notes = f"{notes} — {detail}" if notes else detail
    ds_id = db.create_dataset(role, version, classes, notes or
                              f"Snapshot of {len(items)} labeled examples.", items)
    return db.get_dataset(ds_id)


def _copy_scaled(src: Path, dest: Path) -> None:
    """
    Write *src* to *dest*, downscaling so its longest side is at most
    MATERIALIZE_MAX_SIDE. Falls back to a plain copy if the image cannot be
    decoded, so an odd file never breaks a whole dataset.
    """
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(src) as im:
            if max(im.size) <= MATERIALIZE_MAX_SIDE:
                shutil.copy2(src, dest)
                return
            im = im.convert("RGB")
            im.thumbnail((MATERIALIZE_MAX_SIDE, MATERIALIZE_MAX_SIDE))
            im.save(dest, "JPEG", quality=90)
    except Exception:
        shutil.copy2(src, dest)


def materialize_dataset(dataset_id: int) -> Path:
    """
    Copy a dataset snapshot's items into an ultralytics classify layout,
    downscaled to MATERIALIZE_MAX_SIDE (see the constant for why).
    """
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
        _copy_scaled(src, dest / f"{it['id']:06d}_{src.name}")
    return ds_dir


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _remote():
    """
    Return the remote client if a train server is configured AND reachable,
    else None (so training gracefully runs locally when the box is down).
    """
    from . import remote
    if remote.training_enabled() and remote.reachable():
        return remote
    if remote.training_enabled():
        print("[retrain] remote train server unreachable — falling back to local")
    return None


def _deploy_to_remote(role: str, record: dict) -> None:
    """Push a newly-activated model to the remote inference cache (best effort)."""
    from . import remote
    if not remote.inference_enabled():
        return
    try:
        remote.deploy_model(record["version"], Path(record["path"]))
        print(f"[retrain] deployed {record['version']} to remote inference")
    except Exception as e:
        print(f"[retrain] could not deploy {record['version']} to remote: {e}")


def _base_weights(role: str) -> tuple[str, str]:
    """
    Weights to fine-tune from: the current active model when it is already
    trained on this task (continual learning), otherwise the stock base.
    """
    active = db.get_active_model(role)
    if active and inference.model_ready(role) and Path(active["path"]).exists() \
            and not inference.is_detector(role):
        return active["path"], active["version"]
    # A detector cannot be fine-tuned into a classifier; start from stock.
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
        weights, base_name = _base_weights(role)
        remote = _remote()

        metrics = {"dataset": ds_info, "epochs": epochs, "dataset_version": ds["version"]}
        where = "remote" if (remote and remote.training_enabled()) else "local"
        progress = {"role": role, "version": version, "dataset": ds["version"],
                    "epochs": epochs, "status": "training", "where": where,
                    "counts": ds_info["counts"], "run_id": run_id,
                    "started_at": db.utcnow(), "epoch": 0, "history": []}
        print(f"[retrain] {version} from {ds['version']} ({where}): {ds_info['counts']}")

        if where == "remote":
            def _cb(st):
                # Mirror the box's per-epoch history so the live view can
                # chart loss/accuracy while the run is still going.
                progress.update(
                    epoch=st.get("epoch"),
                    epochs=st.get("epochs") or epochs,
                    remote_status=st.get("status"),
                    history=st.get("history") or [],
                )
            best, remote_metrics = remote.train_remote(
                ds_dir, Path(weights),
                {"role": role, "version": version, "classes": _role_classes(role, _examples(role)),
                 "epochs": epochs, "imgsz": 640},
                progress_cb=_cb,
            )
            metrics.update(remote_metrics)
        else:
            from ultralytics import YOLO

            model = YOLO(weights)
            results = model.train(
                data=str(ds_dir), epochs=epochs, imgsz=640,
                project=str(TRAIN_RUNS_DIR), name=version, exist_ok=True, verbose=True,
            )
            best = Path(results.save_dir) / "weights" / "best.pt"
            if not best.exists():
                raise RuntimeError(f"Training produced no weights at {best}")
            try:
                metrics["top1_accuracy"] = float(results.results_dict.get("metrics/accuracy_top1", 0))
            except Exception:
                pass

        record = registry.register_version(
            role, best, base_model=base_name, metrics=metrics,
            notes=f"Trained on {ds['version']} ({ds_info['total']} examples, {where}).",
            activate=auto_activate,
        )
        if auto_activate:
            inference.invalidate_cache()
            _deploy_to_remote(role, record)

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
