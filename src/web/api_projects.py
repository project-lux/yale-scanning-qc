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
