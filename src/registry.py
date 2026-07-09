"""
Versioned model registry.

Weights live on disk under models/registry/<role>/<version>/model.pt and
are tracked in the ``models`` table. Exactly one version per role is
active at a time; batches record which versions they ran with.
"""

import shutil
from pathlib import Path

from . import db
from .config import (
    BADSCAN_BASE_WEIGHTS,
    PAGETYPE_SEED_WEIGHTS,
    PROJECT_ROOT,
    REGISTRY_DIR,
)


def version_dir(role: str, version: str) -> Path:
    return REGISTRY_DIR / role / version


def store_weights(role: str, version: str, weights_src: Path) -> Path:
    """Copy a weights file into the registry layout and return the new path."""
    dest_dir = version_dir(role, version)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "model.pt"
    shutil.copy2(weights_src, dest)
    return dest


def register_version(role: str, weights_src: Path, base_model: str = "",
                     metrics: dict | None = None, notes: str = "",
                     activate: bool = True) -> dict:
    """Copy weights into the registry and record the new version in the DB."""
    version = db.next_model_version(role)
    dest = store_weights(role, version, weights_src)
    model_id = db.register_model(
        role, version, str(dest), base_model=base_model,
        metrics=metrics, notes=notes, activate=activate,
    )
    return db.get_model(model_id)


def seed_registry() -> None:
    """
    Ensure each role has at least one registered version.

    - badscan: seeded from the yolo26 nano classification base weights.
      Untrained on the QC task, so the pipeline reports 'unknown' until
      the first feedback retrain produces a real binary model.
    - pagetype: seeded from the copied gemini-yolo yolo11n.pt (a stock
      COCO detector — also 'unknown' until a page-type model is trained).
    """
    if not db.list_models("badscan"):
        base = PROJECT_ROOT / BADSCAN_BASE_WEIGHTS
        if base.exists():
            register_version(
                "badscan", base, base_model=BADSCAN_BASE_WEIGHTS,
                notes="Seed: untrained yolo26n-cls base weights. "
                      "Retrain from feedback to produce the first real "
                      "bad_scan/no_bad_scan classifier.",
            )

    if not db.list_models("pagetype"):
        if PAGETYPE_SEED_WEIGHTS.exists():
            register_version(
                "pagetype", PAGETYPE_SEED_WEIGHTS,
                base_model=PAGETYPE_SEED_WEIGHTS.name,
                notes="Seed: copied from gemini-yolo/yolo11n.pt. NOTE: these "
                      "are stock COCO detection weights, not a page-type "
                      "classifier — train a pagetype model to enable page "
                      "classification.",
            )
