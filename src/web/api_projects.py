"""
Project API. A project is one monitored directory; models and datasets stay
global so every project shares the same trained models.
"""

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
    project = db.add_project(str(p), (data.get("name") or "").strip())
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


@bp.get("/<int:project_id>/scans")
def project_scans(project_id: int):
    """Latest scan per file in this project, with its review if any."""
    if db.get_project(project_id) is None:
        return jsonify({"error": "Not found"}), 404
    try:
        per_page = min(500, max(1, int(request.args.get("per_page") or 60)))
        page = max(1, int(request.args.get("page") or 1))
    except ValueError:
        return jsonify({"error": "page and per_page must be integers"}), 400

    rows = db.project_scans(
        project_id,
        flagged=_bool_arg("flagged"),
        reviewed=_bool_arg("reviewed"),
        page_type=(request.args.get("page_type") or None),
        limit=per_page,
        offset=(page - 1) * per_page,
    )
    for r in rows:
        r["filename"] = Path(r["filepath"]).name
        r["has_jpeg"] = bool(r["jpeg_path"]) and Path(r["jpeg_path"]).exists()
    return jsonify({
        "scans": rows,
        "page": page,
        "per_page": per_page,
        "counts": db.project_scan_counts(project_id),
    })


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
        # Whole project when nothing specific is selected.
        ids = [s["id"] for s in db.project_scans(project_id, limit=1_000_000)]
    if not ids:
        return jsonify({"error": "This project has no scans yet"}), 400

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
        if orig.exists() and orig.suffix.lower() in (".jpg", ".jpeg", ".png"):
            src = orig
        else:
            return jsonify({"error": "Preview not available"}), 404
    if request.args.get("full", "").lower() in ("1", "true", "yes", "on"):
        return send_file(src)
    return thumbs.serve("scan", scan_id, src)


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

    from .. import examples
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

    db.save_review(scan_id, verdict, page_type=page_type,
                   notes=(data.get("notes") or "").strip(),
                   reviewer=session.get("user", ""))
    # Make the verdict training data immediately.
    examples.sync("badscan")
    if page_type:
        examples.sync("pagetype")
    return jsonify({"scan": db.get_scan(scan_id), "verdict": verdict})
