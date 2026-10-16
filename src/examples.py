"""
The training-example store: ingest, correct, and serve labeled images.

Before this module a label was implied by which folder a file sat in
(``data/curated/<role>/<label>/``) or was re-derived from a review on every
dataset build. Nothing was addressable, so nothing could be inspected or
corrected in the app. Now every labeled image is a ``training_examples``
row that owns its label, and this module is the only thing that writes them.

Four sources feed in:

  human      — annotator verdicts on scanned files (via ``reviews``). The
               watched folder is a staging area whose files move on within
               days, so a kept verdict's image is copied into the app's own
               store (:mod:`src.preserve`) — and by default only the scans
               the model got wrong are kept at all.
  curated    — real images imported from outside the app (explicit bad
               captures from the share, or dropped into CURATED_DIR by hand)
  project    — presumed-clean pages sampled from a project; the pool the
               dataset builder draws negatives from
  llm        — weak page-type labels from the transcription backend

:func:`sync` is **insert-only and idempotent** for everything but a newer
annotator verdict. It never rewrites the label on a row that was edited
after its review, so a correction made on the training-data page is
durable across re-syncs.
"""

from pathlib import Path

from . import db, preserve
from .config import BAD_SCAN_CLASSES, CURATED_DIR, IMAGES_DIR, MODEL_ROLES

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


def _project_for(path: str, projects: list[dict]) -> dict | None:
    pid = _project_id_for(path, projects)
    return next((p for p in projects if p["id"] == pid), None)


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


# ---------------------------------------------------------------------------
# Annotator verdicts
# ---------------------------------------------------------------------------

def _apply_review(role: str, rv: dict, projects: list[dict], settings: dict) -> str:
    """
    Fold one review (a :func:`db.reviews_with_scans` row) into the store
    for *role*. Returns 'added', 'updated' or 'skipped'.

    A scan that is already in the store is relabeled only when the review
    is newer than the row's last edit — later corrections win. A scan that
    is not yet in the store is kept (copied in with its metadata) when the
    annotator overruled the model, or always when `preserve_agreements` is
    on; otherwise the review stays a review and nothing is kept.
    """
    label, verdict = preserve.label_for(role, rv)
    if label is None:
        return "skipped"
    notes = rv.get("review_notes") or ""
    project = _project_for(rv["filepath"], projects)

    ex = db.training_example_for_scan(role, rv["scan_id"])
    if ex is not None:
        changed = db.apply_review_label_by_id(ex["id"], label, rv["reviewed_at"],
                                              verdict, notes)
        # A weak (LLM) row for this scan lives on the staging path; once an
        # annotator disagrees with the model it is worth keeping for real.
        if not preserve.is_kept(ex["image_path"]) and \
                preserve.should_keep(role, rv, label, settings):
            kept = preserve.keep(rv, project)
            if kept is not None and \
                    db.get_training_example_by_path(role, kept["image_path"]) is None:
                db.update_training_example(
                    ex["id"], image_path=kept["image_path"], meta=kept["meta"],
                    source="human", verdict=verdict)
                changed = True
        return "updated" if changed else "skipped"

    if not preserve.should_keep(role, rv, label, settings):
        return "skipped"
    kept = preserve.keep(rv, project)
    if kept is None:
        return "skipped"   # source already gone — nothing to keep
    new_id = db.add_training_example(
        role=role, label=label, source="human", image_path=kept["image_path"],
        project_id=project["id"] if project else None, scan_id=rv["scan_id"],
        origin_ref=rv["filepath"], notes=notes, verdict=verdict, meta=kept["meta"],
    )
    if new_id is not None:
        return "added"

    # The same image was already kept from an earlier pass of this file:
    # point that row at the newest scan and let the newer review relabel it.
    existing = db.get_training_example_by_path(role, kept["image_path"])
    if existing is None:
        return "skipped"
    if (existing.get("scan_id") or 0) < rv["scan_id"]:
        db.update_training_example(existing["id"], scan_id=rv["scan_id"], meta=kept["meta"])
    changed = db.apply_review_label_by_id(existing["id"], label, rv["reviewed_at"],
                                          verdict, notes)
    return "updated" if changed else "skipped"


def _sync_reviews(role: str, projects: list[dict], settings: dict) -> int:
    added = 0
    for rv in db.reviews_with_scans():
        if _apply_review(role, rv, projects, settings) == "added":
            added += 1
    return added


def _sync_llm_pagetype(projects: list[dict]) -> int:
    """
    Weak page-type labels for unreviewed scans. These are not copied in —
    they point at the staging folder and lapse when it moves on; a human
    answer on the same scan upgrades the row (see :func:`_apply_review`).
    """
    added = 0
    for e in db.llm_pagetype_labels():
        jpeg = e.get("jpeg_path") or e["filepath"]
        if not jpeg or not Path(jpeg).is_file():
            continue
        if db.add_training_example(
            role="pagetype", label=e["label"], source="llm", image_path=jpeg,
            project_id=_project_id_for(e["filepath"], projects),
            scan_id=e["scan_id"], origin_ref=e["filepath"],
        ) is not None:
            added += 1
    return added


def _adopt_external_rows(role: str, projects: list[dict]) -> int:
    """
    One-time (then idempotent) move of older annotator-labeled rows whose
    image still sits in a watched folder into the kept store, so they
    survive that folder churning. Rows whose file is already gone are left
    for :func:`prune_missing`.
    """
    moved = 0
    for ex in db.external_human_examples(role, str(IMAGES_DIR)):
        src = ex["image_path"]
        if not Path(src).is_file():
            continue
        scan = db.latest_scan_for_path(ex.get("origin_ref") or src) \
            or db.latest_scan_for_path(src)
        scan_id = scan["id"] if scan else None
        if scan is not None:
            # Prefer the review-joined row: it carries the pass and model
            # versions as well as the reviewer's answer.
            joined = db.reviews_with_scans(scan_id)
            scan = joined[-1] if joined else db.get_scan(scan_id)
        project = _project_for(ex.get("origin_ref") or src, projects)
        kept = preserve.adopt(src, scan, project)
        if kept is None:
            continue
        if db.get_training_example_by_path(role, kept["image_path"]) is not None:
            continue  # already kept under its hash from another row
        fields = {"image_path": kept["image_path"], "meta": kept["meta"]}
        if scan_id is not None and not ex.get("scan_id"):
            fields["scan_id"] = scan_id
        if project is not None and not ex.get("project_id"):
            fields["project_id"] = project["id"]
        db.update_training_example(ex["id"], **fields)
        moved += 1
    return moved


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def sync(role: str) -> dict:
    """
    Bring the store up to date for *role* from every source.

    New images are inserted. An existing row is only changed when a newer
    annotator verdict relabels it; corrections made on the training-data
    page after that review are kept. Returns per-source counts of newly
    added rows (plus how many older rows were copied into the store).
    """
    projects = db.list_projects()
    settings = db.get_settings()
    adopted = _adopt_external_rows(role, projects)
    added = {
        "human": _sync_reviews(role, projects, settings),
        "curated": _sync_dir(role, CURATED_DIR / role, "curated", projects),
    }
    if role == "pagetype":
        added["llm"] = _sync_llm_pagetype(projects)
    added["total"] = sum(added.values())
    added["adopted"] = adopted
    return added


def sync_all() -> dict:
    return {role: sync(role) for role in MODEL_ROLES}


def sync_scan(scan_id: int, roles: tuple[str, ...] = MODEL_ROLES) -> dict:
    """
    Fold one scan's review into the store right away (the per-verdict
    path, so an annotator's click does not replay every review). Returns
    {role: 'added' | 'updated' | 'skipped'}.
    """
    rows = db.reviews_with_scans(scan_id)
    if not rows:
        return {}
    rv = rows[-1]
    projects = db.list_projects()
    settings = db.get_settings()
    return {role: _apply_review(role, rv, projects, settings) for role in roles}


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


def _derive_verdict(scan: dict, is_bad: bool) -> str:
    flagged = scan.get("bad_scan") == "bad_scan"
    if flagged:
        return "true_positive" if is_bad else "false_positive"
    return "false_negative" if is_bad else "true_negative"


def relabel(example_id: int, label: str, reviewer: str = "") -> dict | None:
    """
    Correct one example's label, refusing classes the role does not define.
    ``original_label`` is left alone so the correction stays visible.

    When the example came from a scan that is still on record, the scan's
    review is updated to match, so the annotator's answer and the training
    label never drift apart. Every correction is logged.
    """
    ex = db.get_training_example(example_id)
    if ex is None:
        return None
    valid = labels_for_role(ex["role"])
    if label not in valid:
        raise ValueError(f"'{label}' is not a {ex['role']} class ({', '.join(valid)})")
    old = ex["label"]
    fields: dict = {"label": label}

    scan = db.get_scan(ex["scan_id"]) if ex.get("scan_id") else None
    if scan is not None:
        if ex["role"] == "badscan":
            verdict = _derive_verdict(scan, label == "bad_scan")
            db.save_review(scan["id"], verdict, notes=scan.get("review_notes") or "",
                           reviewer=reviewer)
            fields["verdict"] = verdict
        else:
            db.save_review_page_type(scan["id"], label, reviewer=reviewer)
    db.update_training_example(example_id, **fields)
    db.log_annotation("relabel", role=ex["role"], scan_id=ex.get("scan_id"),
                      example_id=example_id,
                      filepath=ex.get("origin_ref") or ex["image_path"],
                      old_value=old, new_value=label, reviewer=reviewer)
    return db.get_training_example(example_id)


def history(example_id: int) -> list[dict]:
    """Every logged annotator action that touched this example or its scan."""
    ex = db.get_training_example(example_id)
    if ex is None:
        return []
    return db.list_annotations(example_id=example_id, scan_id=ex.get("scan_id"),
                               filepath=ex.get("origin_ref") or "")


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
