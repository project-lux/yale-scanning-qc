"""
Project API. A project is one monitored directory; models and datasets stay
global so every project shares the same trained models.
"""

import json
from pathlib import Path

from flask import Blueprint, jsonify, request

from .. import db

bp = Blueprint("api_projects", __name__, url_prefix="/api/projects")


@bp.get("")
def list_projects():
    out = []
    for p in db.list_projects():
        row = dict(p)
        row["stats"] = db.project_stats(p["id"])
        row["exists"] = Path(p["path"]).is_dir()
        out.append(row)
    return jsonify({"projects": out})


@bp.post("")
def create_project():
    data = request.get_json(force=True, silent=True) or {}
    path = (data.get("path") or "").strip()
    if not path:
        return jsonify({"error": "A directory path is required"}), 400
    p = Path(path).expanduser()
    if not p.is_dir():
        return jsonify({"error": f"Not a directory: {p}"}), 400
    existing = db.get_project_by_path(str(p))
    if existing:
        return jsonify({"error": "That directory is already a project",
                        "project": existing}), 409
    mode = "tree" if (data.get("mode") == "tree" or data.get("tree")) else "flat"
    project = db.add_project(str(p), (data.get("name") or "").strip(), mode=mode)
    return jsonify({"project": project}), 201


@bp.get("/<int:project_id>")
def get_project(project_id: int):
    p = db.get_project(project_id)
    if p is None:
        return jsonify({"error": "Not found"}), 404
    p["stats"] = db.project_stats(project_id)
    p["exists"] = Path(p["path"]).is_dir()
    return jsonify({"project": p})


@bp.post("/<int:project_id>")
def update_project(project_id: int):
    if db.get_project(project_id) is None:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(force=True, silent=True) or {}
    fields = {}
    if "name" in data:
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"error": "Name cannot be empty"}), 400
        fields["name"] = name
    if "notes" in data:
        fields["notes"] = data.get("notes") or ""
    if "enabled" in data:
        fields["enabled"] = 1 if data["enabled"] else 0
    if not fields:
        return jsonify({"error": "Nothing to change"}), 400
    db.update_project(project_id, **fields)
    return jsonify({"project": db.get_project(project_id)})


@bp.delete("/<int:project_id>")
def delete_project(project_id: int):
    """
    Stop tracking a directory. Scans, batches and training examples are
    kept (examples are just detached), so history and labels survive.
    """
    if db.get_project(project_id) is None:
        return jsonify({"error": "Not found"}), 404
    db.remove_project(project_id)
    return jsonify({"deleted": project_id})


# ---------------------------------------------------------------------------
# Project detail: scans, review, run a pass, reprocess
# ---------------------------------------------------------------------------

def _bool_arg(name: str):
    raw = request.args.get(name)
    if raw is None or raw == "":
        return None
    return raw.lower() in ("1", "true", "yes", "on")


def _subdir_arg(project: dict) -> str | None:
    """Sanitized relative subdirectory filter (tree projects only)."""
    raw = (request.args.get("dir") or "").strip().strip("/")
    if not raw or project.get("mode") != "tree":
        return None
    parts = Path(raw).parts
    if any(part in ("..", "") for part in parts) or Path(raw).is_absolute():
        return None
    return raw


@bp.get("/<int:project_id>/scans")
def project_scans(project_id: int):
    """Latest scan per file in this project, with its review if any."""
    project = db.get_project(project_id)
    if project is None:
        return jsonify({"error": "Not found"}), 404
    try:
        per_page = min(500, max(1, int(request.args.get("per_page") or 60)))
        page = max(1, int(request.args.get("page") or 1))
    except ValueError:
        return jsonify({"error": "page and per_page must be integers"}), 400

    subdir = _subdir_arg(project)
    filters = dict(
        flagged=_bool_arg("flagged"),
        reviewed=_bool_arg("reviewed"),
        page_type=(request.args.get("page_type") or None),
        transcribed=_bool_arg("transcribed"),
        subdir=subdir,
        # The watched folder is a staging area: files leave it for their
        # permanent home. Default to what is actually there.
        present=_bool_arg("present") if "present" in request.args else True,
        kept=_bool_arg("kept"),
    )
    similar_to = request.args.get("similar_to")
    sim_status = None
    if similar_to:
        # Rank by visual similarity to one scan instead of by filename; the
        # filters above still apply to the ranked candidates.
        from .. import similarity

        query = db.get_scan(int(similar_to))
        if query is None:
            return jsonify({"error": "Scan not found"}), 404
        try:
            min_score = float(request.args.get("min_score") or 0)
        except ValueError:
            min_score = 0.0
        ranked = similarity.similar(
            query.get("jpeg_path") or query["filepath"],
            scope=db.project_scan_paths(project_id, subdir),
            limit=min(5000, page * per_page * 4 + 200), min_score=min_score)
        score = dict(ranked)
        rows = db.project_scans(project_id, **filters, scan_ids=list(score),
                                limit=len(score) or 1)
        rows.sort(key=lambda r: -score[r["id"]])
        rows = rows[(page - 1) * per_page: page * per_page]
        for r in rows:
            r["similarity"] = round(score[r["id"]], 4)
        sim_status = similarity.status()
    else:
        rows = db.project_scans(project_id, **filters, limit=per_page,
                                offset=(page - 1) * per_page)
    root = project["path"]
    for r in rows:
        r["filename"] = Path(r["filepath"]).name
        r["kept"] = bool(r.pop("kept_n", 0))
        # A kept copy stands in for the preview once the file has moved on.
        r["has_jpeg"] = (bool(r["jpeg_path"]) and Path(r["jpeg_path"]).exists()) \
            or r["kept"]
        if project.get("mode") == "tree":
            try:
                r["relpath"] = str(Path(r["filepath"]).relative_to(root))
            except ValueError:
                r["relpath"] = r["filename"]
    return jsonify({
        "scans": rows,
        "page": page,
        "per_page": per_page,
        "counts": db.project_scan_counts(project_id, subdir=subdir),
        "similarity": sim_status,
    })


@bp.get("/<int:project_id>/dirs")
def project_dirs(project_id: int):
    """Subdirectories (with scan counts) for the tree project's dir filter."""
    p = db.get_project(project_id)
    if p is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"dirs": db.project_directories(project_id)})


@bp.get("/<int:project_id>/progress")
def project_progress(project_id: int):
    """Whether a pass is running, so the grid can poll while one is active."""
    from .. import pipeline
    return jsonify({
        "running": pipeline.is_running(),
        "progress": pipeline.progress,
    })


@bp.post("/<int:project_id>/run")
def run_project_pass(project_id: int):
    """
    Run a QC pass over just this project, in a worker thread so the request
    returns immediately.
    """
    import threading

    from .. import pipeline

    p = db.get_project(project_id)
    if p is None:
        return jsonify({"error": "Not found"}), 404
    if not Path(p["path"]).is_dir():
        return jsonify({"error": f"Directory is not reachable: {p['path']}"}), 400
    if pipeline.is_running():
        return jsonify({"error": "A pass is already running"}), 409

    data = request.get_json(force=True, silent=True) or {}
    # all=1 re-processes every file; the default is new files only.
    paths = None
    if data.get("all"):
        paths = [str(f) for f in pipeline.discover_files(Path(p["path"]))]
        if not paths:
            return jsonify({"error": "No images found in this directory"}), 400

    def _run():
        try:
            pipeline.run_batch(trigger="manual", directories=[p["path"]],
                               paths=paths, project_id=project_id)
        except Exception as e:  # recorded on the batch row; logged for the console
            print(f"[project-run] {p['name']}: {e}")

    threading.Thread(target=_run, daemon=True,
                     name=f"pass-project-{project_id}").start()
    return jsonify({"started": True, "project_id": project_id,
                    "scope": "all" if paths else "new"}), 202


@bp.post("/<int:project_id>/rerun-stages")
def rerun_project_stages(project_id: int):
    """
    Re-run selected stages over this project's scans with the currently
    active models — the "reprocess" action.
    """
    import threading

    from .. import pipeline

    if db.get_project(project_id) is None:
        return jsonify({"error": "Not found"}), 404
    if pipeline.is_running():
        return jsonify({"error": "A pass is already running"}), 409

    data = request.get_json(force=True, silent=True) or {}
    stages = set(data.get("stages") or ["badscan"])
    unknown = stages - set(pipeline.VALID_STAGES)
    if unknown:
        return jsonify({"error": f"Unknown stage(s): {', '.join(sorted(unknown))}"}), 400

    ids = [int(i) for i in (data.get("scan_ids") or [])]
    if not ids:
        # Whole project when nothing specific is selected — only what is
        # still in the folder; files that moved on cannot be reprocessed.
        ids = [s["id"] for s in db.project_scans(project_id, present=True,
                                                 limit=1_000_000)]
    if not ids:
        return jsonify({"error": "No processed scans are in this folder right now. "
                                 "Run a pass to process new images."}), 400

    def _run():
        try:
            pipeline.rerun_stages(ids, stages,
                                  force_transcribe=bool(data.get("force_transcribe")))
        except Exception as e:
            print(f"[project-rerun] {e}")

    threading.Thread(target=_run, daemon=True,
                     name=f"rerun-project-{project_id}").start()
    return jsonify({"started": True, "scans": len(ids),
                    "stages": sorted(stages)}), 202


@bp.get("/scans/<int:scan_id>/image")
def scan_image(scan_id: int):
    """
    Thumbnail (or ``full=1``) for a scan, from its cached JPEG — the
    originals are TIFFs the browser cannot display.
    """
    from flask import send_file

    from . import thumbs

    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404
    src = Path(scan["jpeg_path"]) if scan["jpeg_path"] else None
    if src is None or not src.exists():
        orig = Path(scan["filepath"])
        kept = db.scan_kept_copy(scan_id)
        if orig.exists() and orig.suffix.lower() in (".jpg", ".jpeg", ".png"):
            src = orig
        elif kept and Path(kept).exists():
            # The file has left the staging folder; the kept copy remains.
            src = Path(kept)
        else:
            return jsonify({"error": "Preview not available"}), 404
    if request.args.get("full", "").lower() in ("1", "true", "yes", "on"):
        return send_file(src)
    return thumbs.serve("scan", scan_id, src)


@bp.post("/<int:project_id>/bulk-verdict")
def bulk_verdict(project_id: int):
    """
    Review many scans at once with the same plain answer — "these are bad
    scans" / "these look fine". Each scan still gets its own TP/FP/FN/TN
    verdict derived from what the model said about it, and every verdict
    becomes training data immediately (same path as a single review).
    """
    from flask import session

    from ..config import all_page_types

    if db.get_project(project_id) is None:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(force=True, silent=True) or {}
    if "is_bad" not in data:
        return jsonify({"error": "is_bad (true/false) is required"}), 400
    ids = [int(i) for i in (data.get("scan_ids") or [])]
    if not ids:
        return jsonify({"error": "scan_ids is required"}), 400
    is_bad = bool(data["is_bad"])

    page_type = (data.get("page_type") or "").strip() or None
    if page_type is not None and page_type not in all_page_types(db.get_settings()):
        return jsonify({"error": f"Invalid page type: {page_type}"}), 400

    reviewer = session.get("user", "")
    notes = (data.get("notes") or "").strip()
    reviewed, kept = 0, 0
    for sid in ids:
        scan = db.get_scan(sid)
        if scan is None:
            continue
        flagged = scan["bad_scan"] == "bad_scan"
        verdict = ("true_positive" if is_bad else "false_positive") if flagged \
            else ("false_negative" if is_bad else "true_negative")
        db.save_review(sid, verdict, page_type=page_type, notes=notes,
                       reviewer=reviewer)
        _log_verdict(scan, verdict, notes, reviewer)
        if page_type and page_type != scan.get("review_page_type"):
            _log_page_type(scan, page_type, reviewer)
        reviewed += 1
        if _keep_scan(sid, page_type is not None):
            kept += 1
    return jsonify({"reviewed": reviewed, "is_bad": is_bad, "kept": kept})


def _log_verdict(scan: dict, verdict: str, notes: str, reviewer: str) -> None:
    """One annotations row per answer, with what it replaced."""
    db.log_annotation("verdict", role="badscan", scan_id=scan["id"],
                      filepath=scan["filepath"],
                      old_value=scan.get("review_verdict") or "",
                      new_value=verdict, notes=notes, reviewer=reviewer)


def _log_page_type(scan: dict, page_type: str | None, reviewer: str) -> None:
    db.log_annotation("page_type", role="pagetype", scan_id=scan["id"],
                      filepath=scan["filepath"],
                      old_value=scan.get("review_page_type") or "",
                      new_value=page_type or "", reviewer=reviewer)


def _keep_scan(scan_id: int, with_page_type: bool) -> bool:
    """
    Fold the scan's review into the training store straight away: a scan
    the model got wrong is copied in with its metadata; one the model got
    right is only recorded as a review. Returns whether anything is kept
    for this scan now.
    """
    from .. import examples

    roles = ("badscan", "pagetype") if with_page_type else ("badscan",)
    examples.sync_scan(scan_id, roles)
    return db.scan_kept_copy(scan_id) is not None


@bp.post("/<int:project_id>/bulk-page-type")
def bulk_page_type(project_id: int):
    """Set one reviewed page type on many scans, leaving bad-scan answers alone."""
    from flask import session

    from .. import examples
    from ..config import all_page_types

    if db.get_project(project_id) is None:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(force=True, silent=True) or {}
    ids = [int(i) for i in (data.get("scan_ids") or [])]
    if not ids:
        return jsonify({"error": "scan_ids is required"}), 400
    page_type = (data.get("page_type") or "").strip()
    if page_type not in all_page_types(db.get_settings()):
        return jsonify({"error": f"Invalid page type: {page_type}"}), 400

    reviewer = session.get("user", "")
    updated = 0
    for sid in ids:
        scan = db.get_scan(sid)
        if scan is None:
            continue
        db.save_review_page_type(sid, page_type, reviewer=reviewer)
        _log_page_type(scan, page_type, reviewer)
        examples.sync_scan(sid, ("pagetype",))
        updated += 1
    return jsonify({"updated": updated, "page_type": page_type})


@bp.get("/scans/<int:scan_id>/transcriptions")
def scan_transcriptions(scan_id: int):
    """Every backend's reading of this scan, for the swap tabs in the viewer."""
    if db.get_scan(scan_id) is None:
        return jsonify({"error": "Scan not found"}), 404
    return jsonify({"transcriptions": db.list_transcriptions(scan_id)})


@bp.get("/scans/<int:scan_id>/labels")
def scan_labels(scan_id: int):
    """Training-data rows made from this scan, with the label each was imported as."""
    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404
    rows = db.training_examples_for_scan(scan["filepath"], scan.get("jpeg_path"),
                                         scan_id=scan_id)
    for r in rows:
        try:
            r["meta"] = json.loads(r.get("meta") or "{}")
        except ValueError:
            r["meta"] = {}
    return jsonify({"examples": rows})


@bp.get("/scans/<int:scan_id>/history")
def scan_history(scan_id: int):
    """Every annotator action on this scan, newest first."""
    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404
    return jsonify({"history": db.list_annotations(scan_id=scan_id,
                                                   filepath=scan["filepath"])})


@bp.post("/scans/<int:scan_id>/page-type")
def scan_page_type(scan_id: int):
    """Set the reviewed page type on its own, without a bad-scan answer."""
    from flask import session

    from .. import examples
    from ..config import all_page_types

    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404
    data = request.get_json(force=True, silent=True) or {}
    page_type = (data.get("page_type") or "").strip() or None
    if page_type is not None and page_type not in all_page_types(db.get_settings()):
        return jsonify({"error": f"Invalid page type: {page_type}"}), 400
    reviewer = session.get("user", "")
    db.save_review_page_type(scan_id, page_type, reviewer=reviewer)
    _log_page_type(scan, page_type, reviewer)
    examples.sync_scan(scan_id, ("pagetype",))
    return jsonify({"scan": db.get_scan(scan_id), "page_type": page_type,
                    "kept": db.scan_kept_copy(scan_id) is not None})


@bp.post("/scans/<int:scan_id>/verdict")
def scan_verdict(scan_id: int):
    """
    Record a review as a plain answer — "is this a bad scan?" — and derive
    the true/false positive/negative bookkeeping from what the model said.

    Reviewers should not have to reason about confusion-matrix terms to
    label an image; the mapping is mechanical, so the app does it:

        model flagged it      + yes -> true_positive
        model flagged it      + no  -> false_positive
        model did not flag it + yes -> false_negative
        model did not flag it + no  -> true_negative
    """
    from flask import session

    from ..config import all_page_types

    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    if "is_bad" not in data:
        return jsonify({"error": "is_bad (true/false) is required"}), 400
    is_bad = bool(data["is_bad"])
    flagged = scan["bad_scan"] == "bad_scan"
    verdict = ("true_positive" if is_bad else "false_positive") if flagged \
        else ("false_negative" if is_bad else "true_negative")

    page_type = (data.get("page_type") or "").strip() or None
    if page_type is not None and page_type not in all_page_types(db.get_settings()):
        return jsonify({"error": f"Invalid page type: {page_type}"}), 400

    reviewer = session.get("user", "")
    notes = (data.get("notes") or "").strip()
    db.save_review(scan_id, verdict, page_type=page_type, notes=notes,
                   reviewer=reviewer)
    _log_verdict(scan, verdict, notes, reviewer)
    if page_type and page_type != scan.get("review_page_type"):
        _log_page_type(scan, page_type, reviewer)
    # Keep it (copy + metadata) right away when the model got it wrong.
    kept = _keep_scan(scan_id, page_type is not None)
    return jsonify({"scan": db.get_scan(scan_id), "verdict": verdict,
                    "kept": kept, "disagreement": verdict in (
                        "false_positive", "false_negative")})
