"""
Training-example API: browse, correct, exclude, and delete labeled images.

This is the surface that was missing — every labeled image is addressable,
so its class can be inspected and changed instead of being implied by a
folder name.
"""

import json
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file, session

from .. import db, examples
from ..config import MODEL_ROLES
from . import thumbs

bp = Blueprint("api_examples", __name__, url_prefix="/api/training-examples")

MODEL_ERRORS = ("false_positive", "false_negative")
AGREEMENTS = ("true_positive", "true_negative")


def _verdicts_arg() -> list[str] | None:
    """
    ?verdict=errors | agreements | <verdict>[,<verdict>] — which review
    outcome the examples came from.
    """
    raw = (request.args.get("verdict") or "").strip()
    if not raw:
        return None
    if raw == "errors":
        return list(MODEL_ERRORS)
    if raw == "agreements":
        return list(AGREEMENTS)
    return [v for v in raw.split(",") if v]


def _decorate(r: dict) -> dict:
    r["exists"] = Path(r["image_path"]).exists()
    r["filename"] = Path(r["image_path"]).name
    r["corrected"] = bool(r["original_label"] and r["label"] != r["original_label"])
    try:
        r["meta"] = json.loads(r.get("meta") or "{}")
    except ValueError:
        r["meta"] = {}
    return r


def _reviewer() -> str:
    return session.get("user", "")


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
    filters = dict(
        role=role,
        label=(request.args.get("label") or None),
        source=(request.args.get("source") or None),
        project_id=(int(project_id) if project_id else None),
        active=_bool_arg("active", True),
        corrected_only=bool(_bool_arg("corrected_only", False)),
        missing_only=bool(_bool_arg("missing_only", False)),
        verdicts=_verdicts_arg(),
    )
    similar_to = request.args.get("similar_to")
    sim_status = None
    if similar_to:
        # Rank by visual similarity to one example; filters still apply.
        from .. import similarity

        query = db.get_training_example(int(similar_to))
        if query is None:
            return jsonify({"error": "Example not found"}), 404
        try:
            min_score = float(request.args.get("min_score") or 0)
        except ValueError:
            min_score = 0.0
        ranked = similarity.similar(
            query["image_path"], scope=db.training_example_paths(role),
            limit=min(5000, page * per_page * 4 + 200), min_score=min_score)
        score = dict(ranked)
        rows = db.list_training_examples(**filters, ids=list(score),
                                         limit=len(score) or 1)
        rows.sort(key=lambda r: -score[r["id"]])
        rows = rows[(page - 1) * per_page: page * per_page]
        for r in rows:
            r["similarity"] = round(score[r["id"]], 4)
        sim_status = similarity.status()
    else:
        rows = db.list_training_examples(**filters, limit=per_page,
                                         offset=(page - 1) * per_page)
    for r in rows:
        _decorate(r)
    return jsonify({
        "role": role,
        "page": page,
        "per_page": per_page,
        "examples": rows,
        "labels": examples.labels_for_role(role),
        "similarity": sim_status,
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
    _decorate(ex)
    ex["labels"] = examples.labels_for_role(ex["role"])
    ex["history"] = examples.history(example_id)
    return jsonify(ex)


@bp.get("/<int:example_id>/history")
def example_history(example_id: int):
    """Every annotator action on this example (or the scan it came from)."""
    if db.get_training_example(example_id) is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"history": examples.history(example_id)})


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

    reviewer = _reviewer()
    if "label" in data:
        try:
            examples.relabel(example_id, (data.get("label") or "").strip(),
                             reviewer=reviewer)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    fields = {k: data[k] for k in ("notes", "active") if k in data}
    if fields:
        db.update_training_example(example_id, **fields)
        where = dict(role=ex["role"], scan_id=ex.get("scan_id"), example_id=example_id,
                     filepath=ex.get("origin_ref") or ex["image_path"], reviewer=reviewer)
        if "notes" in fields and (fields["notes"] or "") != (ex.get("notes") or ""):
            db.log_annotation("notes", notes=fields["notes"] or "", **where)
        if "active" in fields and bool(fields["active"]) != bool(ex.get("active")):
            db.log_annotation("restore" if fields["active"] else "exclude", **where)
    return jsonify({"example": _decorate(db.get_training_example(example_id))})


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

    reviewer = _reviewer()
    n = 0
    if "label" in fields:
        # One at a time so each correction is logged and carried back onto
        # the scan's review, exactly as a single relabel is.
        for i in ids:
            if examples.relabel(i, fields["label"], reviewer=reviewer) is not None:
                n += 1
    if "active" in fields:
        before = {i: db.get_training_example(i) for i in ids}
        n = max(n, db.bulk_update_training_examples(ids, active=fields["active"]))
        for i, ex in before.items():
            if ex is None or bool(ex["active"]) == fields["active"]:
                continue
            db.log_annotation("restore" if fields["active"] else "exclude",
                              role=ex["role"], scan_id=ex.get("scan_id"), example_id=i,
                              filepath=ex.get("origin_ref") or ex["image_path"],
                              reviewer=reviewer)
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
    reviewer = _reviewer()
    for i in ids:
        ex = db.get_training_example(i)
        if ex is not None:
            db.log_annotation("delete", role=ex["role"], scan_id=ex.get("scan_id"),
                              example_id=i,
                              filepath=ex.get("origin_ref") or ex["image_path"],
                              old_value=ex["label"], reviewer=reviewer)
    result = db.delete_training_examples(ids, remove_files=bool(data.get("remove_files")))
    return jsonify(result)


@bp.post("/sync")
def sync_examples():
    """Re-scan every source for labeled images not yet in the store."""
    role = (request.get_json(force=True, silent=True) or {}).get("role")
    if role and role not in MODEL_ROLES:
        return jsonify({"error": f"Unknown role '{role}'"}), 400
    added = examples.sync(role) if role else examples.sync_all()
    from .. import similarity

    similarity.index_missing_async()
    return jsonify({"added": added})


@bp.post("/prune-missing")
def prune_missing():
    """Deactivate examples whose image is gone (e.g. unmounted share)."""
    role = (request.get_json(force=True, silent=True) or {}).get("role") or "badscan"
    if role not in MODEL_ROLES:
        return jsonify({"error": f"Unknown role '{role}'"}), 400
    return jsonify({"deactivated": examples.prune_missing(role)})
