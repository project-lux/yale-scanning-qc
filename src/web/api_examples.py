"""
Training-example API: browse, correct, exclude, and delete labeled images.

This is the surface that was missing — every labeled image is addressable,
so its class can be inspected and changed instead of being implied by a
folder name.
"""

from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from .. import db, examples
from ..config import MODEL_ROLES
from . import thumbs

bp = Blueprint("api_examples", __name__, url_prefix="/api/training-examples")


def _role() -> str:
    role = (request.args.get("role") or "badscan").strip()
    return role if role in MODEL_ROLES else "badscan"


def _bool_arg(name: str, default=None):
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@bp.get("")
def list_examples():
    """Filtered, paged listing for the data browser."""
    role = _role()
    try:
        per_page = min(500, max(1, int(request.args.get("per_page") or 60)))
        page = max(1, int(request.args.get("page") or 1))
    except ValueError:
        return jsonify({"error": "page and per_page must be integers"}), 400

    project_id = request.args.get("project_id")
    rows = db.list_training_examples(
        role=role,
        label=(request.args.get("label") or None),
        source=(request.args.get("source") or None),
        project_id=(int(project_id) if project_id else None),
        active=_bool_arg("active", True),
        corrected_only=bool(_bool_arg("corrected_only", False)),
        missing_only=bool(_bool_arg("missing_only", False)),
        limit=per_page,
        offset=(page - 1) * per_page,
    )
    for r in rows:
        r["exists"] = Path(r["image_path"]).exists()
        r["filename"] = Path(r["image_path"]).name
        r["corrected"] = bool(r["original_label"] and r["label"] != r["original_label"])
    return jsonify({
        "role": role,
        "page": page,
        "per_page": per_page,
        "examples": rows,
        "labels": examples.labels_for_role(role),
    })


@bp.get("/stats")
def stats():
    role = _role()
    out = db.training_example_stats(role)
    out["role"] = role
    out["labels"] = examples.labels_for_role(role)
    out["projects"] = [
        {"id": p["id"], "name": p["name"]} for p in db.list_projects()
    ]
    return jsonify(out)


@bp.get("/<int:example_id>")
def get_example(example_id: int):
    ex = db.get_training_example(example_id)
    if ex is None:
        return jsonify({"error": "Not found"}), 404
    ex["exists"] = Path(ex["image_path"]).exists()
    ex["filename"] = Path(ex["image_path"]).name
    ex["corrected"] = bool(ex["original_label"] and ex["label"] != ex["original_label"])
    ex["labels"] = examples.labels_for_role(ex["role"])
    return jsonify(ex)


@bp.get("/<int:example_id>/image")
def example_image(example_id: int):
    """Serve the image — a cached thumbnail unless ``full=1``."""
    ex = db.get_training_example(example_id)
    if ex is None:
        return jsonify({"error": "Not found"}), 404
    src = Path(ex["image_path"])
    if not src.exists():
        return jsonify({"error": "Image file is missing"}), 404

    if _bool_arg("full", False):
        return send_file(src)
    return thumbs.serve("example", example_id, src)


@bp.post("/<int:example_id>")
def update_example(example_id: int):
    """Correct a label, edit notes, or include/exclude from training."""
    data = request.get_json(force=True, silent=True) or {}
    ex = db.get_training_example(example_id)
    if ex is None:
        return jsonify({"error": "Not found"}), 404

    if "label" in data:
        try:
            examples.relabel(example_id, (data.get("label") or "").strip())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    fields = {k: data[k] for k in ("notes", "active") if k in data}
    if fields:
        db.update_training_example(example_id, **fields)
    return jsonify({"example": db.get_training_example(example_id)})


@bp.post("/bulk")
def bulk_update():
    """Relabel or include/exclude many examples at once."""
    data = request.get_json(force=True, silent=True) or {}
    ids = [int(i) for i in (data.get("ids") or [])]
    if not ids:
        return jsonify({"error": "No examples selected"}), 400

    fields = {}
    if "label" in data:
        label = (data.get("label") or "").strip()
        # Every selected example must accept the class, or we change nothing.
        roles = {db.get_training_example(i)["role"]
                 for i in ids if db.get_training_example(i)}
        for role in roles:
            if label not in examples.labels_for_role(role):
                return jsonify({
                    "error": f"'{label}' is not a {role} class"
                }), 400
        fields["label"] = label
    if "active" in data:
        fields["active"] = bool(data["active"])
    if not fields:
        return jsonify({"error": "Nothing to change"}), 400

    n = db.bulk_update_training_examples(ids, **fields)
    return jsonify({"updated": n})


@bp.post("/delete")
def delete_examples():
    """
    Permanently remove rows. ``remove_files`` also unlinks the image, but
    only for app-produced copies (synthetic/curated) — never a reviewer's
    original scan on the share.
    """
    data = request.get_json(force=True, silent=True) or {}
    ids = [int(i) for i in (data.get("ids") or [])]
    if not ids:
        return jsonify({"error": "No examples selected"}), 400
    result = db.delete_training_examples(ids, remove_files=bool(data.get("remove_files")))
    return jsonify(result)


@bp.post("/sync")
def sync_examples():
    """Re-scan every source for labeled images not yet in the store."""
    role = (request.get_json(force=True, silent=True) or {}).get("role")
    if role and role not in MODEL_ROLES:
        return jsonify({"error": f"Unknown role '{role}'"}), 400
    added = examples.sync(role) if role else examples.sync_all()
    return jsonify({"added": added})


@bp.post("/prune-missing")
def prune_missing():
    """Deactivate examples whose image is gone (e.g. unmounted share)."""
    role = (request.get_json(force=True, silent=True) or {}).get("role") or "badscan"
    if role not in MODEL_ROLES:
        return jsonify({"error": f"Unknown role '{role}'"}), 400
    return jsonify({"deactivated": examples.prune_missing(role)})
