"""
The training-example store: ingest, correct, and serve labeled images.

Before this module a label was implied by which folder a file sat in
(``data/curated/<role>/<label>/``) or was re-derived from a review on every
dataset build. Nothing was addressable, so nothing could be inspected or
corrected in the app. Now every labeled image is a ``training_examples``
row that owns its label, and this module is the only thing that writes them.

Three sources feed in:

  human      — reviewer verdicts on scanned files (via ``reviews``)
  curated    — real images imported from outside the app (explicit bad
               captures from the share, or dropped into CURATED_DIR by hand)
  project    — presumed-clean pages sampled from a project; the pool the
               dataset builder draws negatives from
  llm        — weak page-type labels from the transcription backend

:func:`sync` is **insert-only and idempotent**. It never rewrites the label
on a row that already exists, so re-running it after someone has corrected
a label leaves that correction alone. That is the whole reason corrections
are durable.
"""

from pathlib import Path

from . import db
from .config import BAD_SCAN_CLASSES, CURATED_DIR, MODEL_ROLES

# Sources that live as files the app itself wrote, and so may be deleted
# from disk when their row is deleted.
OWNED_SOURCES = ("curated", "project")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _project_id_for(path: str, projects: list[dict]) -> int | None:
    """
    The project whose directory contains *path*. Longest matching path wins,
    so a nested project beats its parent.
    """
    best, best_len = None, -1
    for p in projects:
        root = p["path"].rstrip("/")
        if (path == root or path.startswith(root + "/")) and len(root) > best_len:
            best, best_len = p["id"], len(root)
    return best


def _sync_dir(role: str, base: Path, source: str, projects: list[dict]) -> int:
    """Ingest <base>/<label>/*.jpg as *source* examples for *role*."""
    added = 0
    if not base.is_dir():
        return added
    for label_dir in sorted(base.iterdir()):
        # A leading underscore marks staging/retired directories.
        if not label_dir.is_dir() or label_dir.name.startswith("_"):
            continue
        for img in sorted(label_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if db.add_training_example(
                role=role, label=label_dir.name, source=source,
                image_path=str(img),
                project_id=_project_id_for(str(img), projects),
            ) is not None:
                added += 1
    return added


def _sync_reviews(role: str, projects: list[dict]) -> int:
    """
    Ingest reviewer verdicts (and LLM page types) as examples.

    The verdict is stored on the row so the dataset builder can tell which
    examples a previous model got wrong and always keep them.
    """
    if role == "badscan":
        rows = db.labeled_badscan_examples()
    else:
        rows = db.labeled_pagetype_examples()

    added = 0
    for e in rows:
        jpeg = e.get("jpeg_path") or ""
        if not jpeg:
            continue
        source = e.get("source") or "human"
        if source.startswith("llm"):
            source = "llm"
        if db.add_training_example(
            role=role, label=e["label"], source=source, image_path=jpeg,
            project_id=_project_id_for(e.get("filepath") or jpeg, projects),
            origin_ref=e.get("filepath") or "",
            verdict=e.get("verdict") or "",
        ) is not None:
            added += 1
    return added


def sync(role: str) -> dict:
    """
    Bring the store up to date for *role* from every source.

    Insert-only: existing rows (and any corrections made to them) are left
    untouched. Returns per-source counts of newly added rows.
    """
    projects = db.list_projects()
    added = {
        "human": _sync_reviews(role, projects),
        "curated": _sync_dir(role, CURATED_DIR / role, "curated", projects),
    }
    added["total"] = sum(added.values())
    return added


def sync_all() -> dict:
    return {role: sync(role) for role in MODEL_ROLES}


def for_training(role: str) -> list[dict]:
    """
    Active examples for *role*, shaped the way :mod:`src.retrain` expects
    and filtered to images that still exist on disk.
    """
    out = []
    for r in db.list_training_examples(role=role, active=True, limit=1_000_000):
        p = r["image_path"]
        if not p or not Path(p).exists():
            continue
        out.append({
            "id": r["id"],
            "filepath": p,
            "jpeg_path": p,
            "label": r["label"],
            "source": r["source"],
            "project_id": r["project_id"],
            "verdict": r["verdict"] or "",
        })
    return out


def labels_for_role(role: str) -> list[str]:
    """Classes offered when relabeling, so the UI never invents a class."""
    if role == "badscan":
        return list(BAD_SCAN_CLASSES)
    from .config import all_page_types
    return all_page_types(db.get_settings())


def relabel(example_id: int, label: str) -> dict | None:
    """
    Correct one example's label, refusing classes the role does not define.
    ``original_label`` is left alone so the correction stays visible.
    """
    ex = db.get_training_example(example_id)
    if ex is None:
        return None
    valid = labels_for_role(ex["role"])
    if label not in valid:
        raise ValueError(f"'{label}' is not a {ex['role']} class ({', '.join(valid)})")
    db.update_training_example(example_id, label=label)
    return db.get_training_example(example_id)


def prune_missing(role: str) -> int:
    """
    Deactivate examples whose image has disappeared (an unmounted share, a
    deleted derivative). Soft, so they come back if the file returns.
    """
    gone = [r["id"] for r in db.list_training_examples(
        role=role, active=True, limit=1_000_000)
        if not Path(r["image_path"]).exists()]
    if gone:
        db.bulk_update_training_examples(gone, active=False)
    return len(gone)
