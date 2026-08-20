"""
Flask application setup for the Yale Scanning QC workflow.

A scheduler (or a manual pass) sweeps each project's directory, runs the
active model versions over new scans, and stores everything in SQLite.
Reviewers triage flagged scans; their verdicts become training data for the
next model version.

The interface lives in :mod:`src.web` — blueprints with Jinja templates
under ``src/templates`` and per-page JS under ``src/static``. This module
keeps app construction, authentication, and the shared JSON API.
"""

import os
import secrets
import threading
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv(Path(__file__).parent.parent / ".env")

from . import db, pipeline, registry, retrain, scheduler
from .config import (
    DEFAULT_SETTINGS,
    BAD_SCAN_CLASSES,
    PAGE_TYPES,
    SCAN_EXTENSIONS,
    all_page_types,
)

app = Flask(__name__)

# The project-centric interface lives in src/web (blueprints) with real
# templates/static under src/. The classic single-page UI below is untouched
# and still served at "/" while the new surfaces are built out.
from . import web  # noqa: E402  (imported after app for blueprint registration)

web.register(app)


# ---------------------------------------------------------------------------
# Authentication (unchanged from the original app)
# ---------------------------------------------------------------------------

_AUTH_USERNAME = (os.environ.get("ADMIN_USERNAME") or "admin").strip()
_AUTH_PASSWORD_PLAIN = (os.environ.get("ADMIN_PASSWORD") or "").strip()

if not _AUTH_PASSWORD_PLAIN:
    raise RuntimeError(
        "ADMIN_PASSWORD is not set. Copy .env.example to .env and set "
        "ADMIN_USERNAME / ADMIN_PASSWORD before starting the app."
    )

_AUTH_PWHASH = generate_password_hash(_AUTH_PASSWORD_PLAIN)
del _AUTH_PASSWORD_PLAIN

app.secret_key = (os.environ.get("SECRET_KEY") or "").strip() or secrets.token_hex(32)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.permanent_session_lifetime = 60 * 60 * 12  # 12 hours

_AUTH_EXEMPT_ENDPOINTS = {"login", "logout", "static"}


def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Not authenticated"}), 401
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapped


@app.before_request
def _require_login():
    if request.endpoint in _AUTH_EXEMPT_ENDPOINTS:
        return None
    if session.get("user"):
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not authenticated"}), 401
    return redirect(url_for("login", next=request.path))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

_initialized = False
_init_lock = threading.Lock()


def init_app() -> None:
    """Create the DB, seed the model registry, start the scheduler."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        db.init_db()
        registry.seed_registry()
        scheduler.start()
        _initialized = True
        print("Yale Scanning QC (batch mode) initialized.")


init_app()


# ---------------------------------------------------------------------------
# Routes: auth + index
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        if username == _AUTH_USERNAME and check_password_hash(_AUTH_PWHASH, password):
            session.clear()
            session["user"] = username
            session.permanent = True
            nxt = request.args.get("next") or request.form.get("next") or url_for("index")
            if not nxt.startswith("/"):
                nxt = url_for("index")
            return redirect(nxt)
        error = "Invalid username or password."

    nxt = request.args.get("next", "")
    return render_template_string(LOGIN_TEMPLATE, error=error, next=nxt)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))




# ---------------------------------------------------------------------------
# API: global state
# ---------------------------------------------------------------------------

@app.route("/api/state")
def api_state():
    from . import inference, remote
    settings = db.get_settings()
    return jsonify({
        "settings": settings,
        "directories": db.list_directories(),
        "active_models": {
            "badscan": db.get_active_model("badscan"),
            "pagetype": db.get_active_model("pagetype"),
        },
        "model_ready": {
            "badscan": inference.model_ready("badscan"),
            "pagetype": inference.model_ready("pagetype"),
        },
        "batch_running": pipeline.is_running(),
        "batch_progress": pipeline.progress,
        "train_running": retrain.is_running(),
        "train_progress": retrain.progress,
        "remote": remote.status(),
        "next_scheduled": scheduler.next_run_description(),
        "page_types": all_page_types(settings),
        "builtin_page_types": PAGE_TYPES,
        "pagetype_model_classes": sorted(inference.model_classes("pagetype") or []),
        "badscan_classes": BAD_SCAN_CLASSES,
    })


# ---------------------------------------------------------------------------
# API: batches + scans
# ---------------------------------------------------------------------------

@app.route("/api/batches")
def api_batches():
    return jsonify({"batches": db.list_batches()})


@app.route("/api/batches/preview", methods=["POST"])
def api_preview_batch():
    """Dry-run scan of the monitored directories: image counts, no processing."""
    if not db.list_directories(enabled_only=True):
        return jsonify({"error": "No monitored directories configured (see Settings)"}), 400
    return jsonify(pipeline.preview_work())


@app.route("/api/batches/run", methods=["POST"])
def api_run_batch():
    if pipeline.is_running():
        return jsonify({"error": "A batch is already running"}), 409
    if not db.list_directories(enabled_only=True):
        return jsonify({"error": "No monitored directories configured (see Settings)"}), 400

    data = request.get_json(force=True) if request.is_json else {}
    known = {d["path"] for d in db.list_directories(enabled_only=True)}

    directories = data.get("directories")
    if directories is not None:
        directories = [d for d in directories if d in known]
        if not directories:
            return jsonify({"error": "No valid directories selected"}), 400

    # Explicit file list (folder-/image-level run from the Directories view).
    paths = data.get("paths")
    if paths is not None:
        roots = [Path(k) for k in known]
        paths = [p for p in paths
                 if any(Path(p).is_relative_to(r) for r in roots)]
        if not paths:
            return jsonify({"error": "No valid files selected"}), 400

    kwargs = {"trigger": "manual", "directories": directories, "paths": paths}
    for key, role in (("badscan_model_id", "badscan"), ("pagetype_model_id", "pagetype")):
        val = data.get(key)
        if val is not None:
            record = db.get_model(int(val))
            if record is None or record["role"] != role:
                return jsonify({"error": f"Invalid {key}"}), 400
            kwargs[key] = record["id"]

    threading.Thread(target=pipeline.run_batch, kwargs=kwargs, daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/batches/cancel", methods=["POST"])
def api_cancel_batch():
    if not pipeline.cancel():
        return jsonify({"error": "No batch is running"}), 409
    return jsonify({"cancelling": True})


@app.route("/api/batches/<int:batch_id>", methods=["DELETE"])
def api_delete_batch(batch_id: int):
    batch = db.get_batch(batch_id)
    if batch is None:
        return jsonify({"error": "Batch not found"}), 404
    if pipeline.is_running() and pipeline.progress.get("batch_id") == batch_id:
        return jsonify({"error": "This batch is currently running — cancel it first"}), 409
    removed = db.delete_batch(batch_id)
    return jsonify({"deleted": batch["label"], **removed})


@app.route("/api/batches/<int:batch_id>/reviews-count")
def api_batch_reviews_count(batch_id: int):
    return jsonify({"reviews": db.count_batch_reviews(batch_id)})


def _empty_node(name: str, rel: str) -> dict:
    return {"name": name, "rel": rel, "dirs": {}, "files": []}


def _finalize_node(node: dict) -> dict:
    """Sort children, convert dir map to list, compute counts bottom-up."""
    node["dirs"] = [_finalize_node(d) for _, d in sorted(node["dirs"].items(),
                                                         key=lambda kv: kv[0].lower())]
    node["files"].sort(key=lambda f: f["name"].lower())
    node["total"] = len(node["files"]) + sum(d["total"] for d in node["dirs"])
    node["new"] = (sum(1 for f in node["files"] if not f["processed"])
                   + sum(d["new"] for d in node["dirs"]))
    return node


@app.route("/api/directories/listing")
def api_directories_listing():
    """
    Nested tree of each monitored directory mirroring the real folder
    structure, with per-file processing state (latest scan, which stages
    have output) for the Directories view.
    """
    latest = db.latest_scans_by_path()
    out = []
    for d in db.list_directories(enabled_only=True):
        root = Path(d["path"])
        tree = _empty_node(root.name or str(root), ".")
        for f in pipeline.discover_files(root):
            try:
                rel_parts = f.relative_to(root).parts
            except ValueError:
                rel_parts = (f.name,)
            cur = tree
            so_far: list[str] = []
            for part in rel_parts[:-1]:
                so_far.append(part)
                cur = cur["dirs"].setdefault(part, _empty_node(part, "/".join(so_far)))
            scan = latest.get(str(f))
            cur["files"].append({
                "name": f.name,
                "path": str(f),
                "processed": scan is not None,
                "scan_id": scan["id"] if scan else None,
                "batch_id": scan["batch_id"] if scan else None,
                "m": bool(scan and scan["bad_scan"]),
                "p": bool(scan and scan["page_type"]),
                "t": bool(scan and scan["transcribed"]),
                "error": bool(scan and scan["status"] == "error"),
            })
        out.append({"id": d["id"], "path": d["path"], "tree": _finalize_node(tree)})
    return jsonify({"directories": out})


def _parse_stages(data: dict) -> set[str]:
    return {s for s in (data.get("stages") or []) if s in pipeline.VALID_STAGES}


@app.route("/api/scans/<int:scan_id>/rerun-stages", methods=["POST"])
def api_scan_rerun_stages(scan_id: int):
    """Re-run selected stages on one scan, in place (synchronous)."""
    if pipeline.is_running():
        return jsonify({"error": "A batch or re-run is already in progress"}), 409
    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404

    data = request.get_json(force=True)
    stages = _parse_stages(data)
    if not stages:
        return jsonify({"error": "No valid stages given"}), 400

    result = pipeline.rerun_stages(
        [scan_id], stages,
        force_transcribe=bool(data.get("force_transcribe")) and "transcription" in stages,
    )
    if result is None:
        return jsonify({"error": "A batch or re-run is already in progress"}), 409
    return jsonify({"scan": db.get_scan(scan_id), **result})


@app.route("/api/batches/<int:batch_id>/rerun-stages", methods=["POST"])
def api_batch_rerun_stages(batch_id: int):
    """Re-run selected stages over every scan in a batch (background)."""
    if pipeline.is_running():
        return jsonify({"error": "A batch or re-run is already in progress"}), 409
    if db.get_batch(batch_id) is None:
        return jsonify({"error": "Batch not found"}), 404

    stages = _parse_stages(request.get_json(force=True))
    if not stages:
        return jsonify({"error": "No valid stages given"}), 400

    scan_ids = [s["id"] for s in db.list_scans(batch_id)]
    if not scan_ids:
        return jsonify({"error": "Batch has no scans"}), 400

    threading.Thread(target=pipeline.rerun_stages, args=(scan_ids, stages),
                     daemon=True).start()
    return jsonify({"started": True, "scans": len(scan_ids), "stages": sorted(stages)})


@app.route("/api/batches/<int:batch_id>/rerun", methods=["POST"])
def api_rerun_batch(batch_id: int):
    if pipeline.is_running():
        return jsonify({"error": "A batch is already running"}), 409
    batch = db.get_batch(batch_id)
    if batch is None:
        return jsonify({"error": "Batch not found"}), 404
    threading.Thread(
        target=pipeline.run_batch,
        kwargs={"trigger": "rerun", "rerun_of": batch_id},
        daemon=True,
    ).start()
    return jsonify({"started": True})


@app.route("/api/batches/<int:batch_id>")
def api_batch_detail(batch_id: int):
    batch = db.get_batch(batch_id)
    if batch is None:
        return jsonify({"error": "Batch not found"}), 404
    return jsonify({"batch": batch, "scans": db.list_scans(batch_id)})


@app.route("/api/scans/<int:scan_id>")
def api_scan_get(scan_id: int):
    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404
    return jsonify({"scan": scan})


@app.route("/api/files/preview")
def api_file_preview():
    """
    Preview any image under a monitored directory, processed or not.
    Nothing is stored in the database — TIFFs are converted on demand
    into the on-disk qc/cache next to the scans, other formats are
    served directly from disk.
    """
    raw = (request.args.get("path") or "").strip()
    if not raw:
        return jsonify({"error": "No path given"}), 400
    filepath = Path(raw)
    roots = [Path(d["path"]) for d in db.list_directories()]
    if not any(filepath.is_relative_to(r) for r in roots):
        return jsonify({"error": "Path is not under a monitored directory"}), 403
    if not filepath.exists() or filepath.suffix.lower() not in SCAN_EXTENSIONS:
        return jsonify({"error": "Not a previewable image"}), 404

    root = next(r for r in sorted(roots, key=lambda x: len(str(x)), reverse=True)
                if filepath.is_relative_to(r))
    try:
        jp = pipeline._ensure_jpeg(root, filepath)
        return send_file(jp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bad-scans")
def api_bad_scans():
    """All current bad scans across passes — the Bad Scan review queue."""
    return jsonify({"scans": db.list_bad_scans()})


@app.route("/api/scans/<int:scan_id>/versions")
def api_scan_versions(scan_id: int):
    """All pass versions for the file this scan belongs to."""
    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404
    return jsonify({"versions": db.scans_for_path(scan["filepath"])})


@app.route("/api/scans/<int:scan_id>/preview")
def api_scan_preview(scan_id: int):
    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404
    jp = Path(scan["jpeg_path"]) if scan["jpeg_path"] else None
    if jp is None or not jp.exists():
        # Fall back to the original file if it is directly viewable.
        orig = Path(scan["filepath"])
        if orig.exists() and orig.suffix.lower() in (".jpg", ".jpeg", ".png"):
            return send_file(orig)
        return jsonify({"error": "Preview not available"}), 404
    return send_file(jp, mimetype="image/jpeg")


@app.route("/api/scans/<int:scan_id>/review", methods=["POST"])
def api_review_scan(scan_id: int):
    scan = db.get_scan(scan_id)
    if scan is None:
        return jsonify({"error": "Scan not found"}), 404

    data = request.get_json(force=True)
    verdict = (data.get("verdict") or "").strip()
    valid = {"true_positive", "false_positive", "false_negative", "true_negative"}
    if verdict not in valid:
        return jsonify({"error": f"Invalid verdict (expected one of {sorted(valid)})"}), 400

    page_type = (data.get("page_type") or "").strip() or None
    if page_type is not None and page_type not in all_page_types(db.get_settings()):
        return jsonify({"error": f"Invalid page type: {page_type}"}), 400

    db.save_review(
        scan_id, verdict,
        page_type=page_type,
        notes=(data.get("notes") or "").strip(),
        reviewer=session.get("user", ""),
    )
    # Fold the verdict into the training-example store straight away so it
    # is visible (and correctable) in the data browser without waiting for
    # the next dataset build.
    from . import examples as _examples
    _examples.sync("badscan")
    if page_type:
        _examples.sync("pagetype")
    return jsonify({"scan": db.get_scan(scan_id)})


# ---------------------------------------------------------------------------
# API: models + training
# ---------------------------------------------------------------------------

@app.route("/api/models")
def api_models():
    return jsonify({
        "models": db.list_models(),
        "training_runs": db.list_training_runs(),
        "datasets": {
            "badscan": retrain.dataset_summary("badscan"),
            "pagetype": retrain.dataset_summary("pagetype"),
        },
        "dataset_versions": db.list_datasets(),
        "train_running": retrain.is_running(),
        "train_progress": retrain.progress,
    })


@app.route("/api/models/<int:model_id>/activate", methods=["POST"])
def api_activate_model(model_id: int):
    from . import inference
    record = db.activate_model(model_id)
    if record is None:
        return jsonify({"error": "Model not found"}), 404
    inference.invalidate_cache()
    return jsonify({"activated": record["version"]})


@app.route("/api/train", methods=["POST"])
def api_train():
    if retrain.is_running():
        return jsonify({"error": "A training run is already in progress"}), 409
    data = request.get_json(force=True)
    role = (data.get("role") or "").strip()
    if role not in ("badscan", "pagetype"):
        return jsonify({"error": "role must be 'badscan' or 'pagetype'"}), 400

    settings = db.get_settings()
    dataset_id = data.get("dataset_id")
    if dataset_id is not None:
        ds = db.get_dataset(int(dataset_id))
        if ds is None or ds["role"] != role:
            return jsonify({"error": "Invalid dataset for this role"}), 400
        dataset_id = int(dataset_id)
    else:
        summary = retrain.dataset_summary(role)
        if not summary["trainable"]:
            return jsonify({
                "error": "Not enough labeled examples to train: "
                         f"{summary['counts']} (need >= {summary['min_per_class']} per class)"
            }), 400

    epochs = int(data.get("epochs") or settings.get("retrain_epochs", "50"))
    auto_activate = bool(data.get("auto_activate",
                                  settings.get("retrain_auto_activate", "1") == "1"))
    retrain.train_async(role, epochs=epochs, auto_activate=auto_activate, dataset_id=dataset_id)
    return jsonify({"started": True, "role": role, "epochs": epochs})


@app.route("/api/datasets", methods=["GET"])
def api_datasets():
    return jsonify({"datasets": db.list_datasets()})


@app.route("/api/remote/status")
def api_remote_status():
    from . import remote
    return jsonify(remote.status(refresh=True))


@app.route("/api/datasets/build", methods=["POST"])
def api_build_dataset():
    data = request.get_json(force=True)
    role = (data.get("role") or "").strip()
    if role not in ("badscan", "pagetype"):
        return jsonify({"error": "role must be 'badscan' or 'pagetype'"}), 400
    try:
        ds = retrain.build_dataset(role, notes=(data.get("notes") or "").strip())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"dataset": ds})


@app.route("/api/datasets/<int:dataset_id>")
def api_dataset_detail(dataset_id: int):
    ds = db.get_dataset(dataset_id)
    if ds is None:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify({"dataset": ds, "items": db.dataset_items(dataset_id)})


@app.route("/api/datasets/<int:dataset_id>", methods=["DELETE"])
def api_delete_dataset(dataset_id: int):
    if db.get_dataset(dataset_id) is None:
        return jsonify({"error": "Dataset not found"}), 404
    db.delete_dataset(dataset_id)
    return jsonify({"deleted": dataset_id})


@app.route("/api/dataset-items/<int:item_id>/image")
def api_dataset_item_image(item_id: int):
    """Serve a dataset item's image (for the dataset browser)."""
    with db.get_db() as conn:
        row = conn.execute("SELECT image_path FROM dataset_items WHERE id=?", (item_id,)).fetchone()
    if row is None or not Path(row["image_path"]).exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(row["image_path"])


# ---------------------------------------------------------------------------
# API: settings + directories
# ---------------------------------------------------------------------------

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify({"settings": db.get_settings()})
    data = request.get_json(force=True)
    allowed = set(DEFAULT_SETTINGS.keys())
    updates = {k: str(v) for k, v in data.items() if k in allowed}
    changed_classes = ("custom_page_types" in updates
                       and updates["custom_page_types"] != db.get_setting("custom_page_types"))
    db.set_settings(updates)
    if changed_classes:
        # Page-type model validity depends on the configured class list.
        from . import inference
        inference.invalidate_cache()
    return jsonify({"settings": db.get_settings()})


@app.route("/api/directories", methods=["POST"])
def api_add_directory():
    data = request.get_json(force=True)
    path = (data.get("path") or "").strip()
    if not path:
        return jsonify({"error": "No path provided"}), 400
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return jsonify({"error": f"Not a valid directory: {p}"}), 400
    db.add_directory(str(p))
    return jsonify({"directories": db.list_directories()})


@app.route("/api/directories/<int:dir_id>", methods=["DELETE"])
def api_remove_directory(dir_id: int):
    db.remove_directory(dir_id)
    return jsonify({"directories": db.list_directories()})


@app.route("/api/directories/<int:dir_id>/toggle", methods=["POST"])
def api_toggle_directory(dir_id: int):
    data = request.get_json(force=True)
    db.set_directory_enabled(dir_id, bool(data.get("enabled", True)))
    return jsonify({"directories": db.list_directories()})


@app.route("/api/browse", methods=["POST"])
def api_browse():
    """List directories under a given path for the in-browser folder picker."""
    import platform

    data = request.get_json(force=True) if request.is_json else {}
    req_path = data.get("path", "").strip()

    if not req_path:
        if platform.system() == "Windows":
            req_path = os.environ.get("USERPROFILE", "C:\\")
        else:
            req_path = os.path.expanduser("~")

    p = Path(req_path).expanduser().resolve()
    if not p.is_dir():
        return jsonify({"error": f"Not a directory: {p}"}), 400

    dirs = []
    try:
        for entry in sorted(os.scandir(p), key=lambda e: e.name.lower()):
            if entry.is_dir() and not entry.name.startswith("."):
                dirs.append({"name": entry.name, "path": str(p / entry.name)})
    except PermissionError:
        pass

    image_count = sum(
        1 for entry in os.scandir(p)
        if entry.is_file() and Path(entry.name).suffix.lower() in SCAN_EXTENSIONS
    )

    return jsonify({
        "current": str(p),
        "parent": str(p.parent) if p != p.parent else None,
        "dirs": dirs,
        "tiff_count": image_count,
    })


# ---------------------------------------------------------------------------
# Login Template (unchanged design)
# ---------------------------------------------------------------------------
LOGIN_TEMPLATE = r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sign in · Yale Scanning QC</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root {
    --yale-blue:#00356b;
    --yale-blue-dark:#002654;
    --yale-blue-lighter:#286dc0;
    --gray-1:#222;
    --gray-3:#777;
    --gray-4:#dcdcdc;
    --gray-5:#f5f5f5;
    --white:#fff;
    --error:#c0392b;
}
* { box-sizing:border-box; margin:0; padding:0; }
body {
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    background:linear-gradient(135deg, var(--yale-blue) 0%, var(--yale-blue-dark) 100%);
    color:var(--gray-1);
    min-height:100vh;
    display:flex; align-items:center; justify-content:center;
    padding:1.5rem;
}
.login-card {
    background:var(--white);
    border-radius:14px;
    box-shadow:0 18px 50px rgba(0,0,0,.35);
    width:100%; max-width:380px;
    padding:2.25rem 2rem 2rem;
}
.brand { text-align:center; margin-bottom:1.5rem; }
.brand-mark {
    display:inline-flex; align-items:center; justify-content:center;
    width:48px; height:48px; border-radius:12px;
    background:var(--yale-blue); color:var(--white);
    font-weight:700; font-size:1.1rem; letter-spacing:.04em;
    margin-bottom:.75rem;
}
.brand h1 { font-size:1.05rem; font-weight:600; color:var(--gray-1); letter-spacing:.01em; }
.brand p { font-size:.78rem; color:var(--gray-3); margin-top:.2rem; }
.field { margin-bottom:1rem; }
.field label {
    display:block; font-size:.72rem; text-transform:uppercase;
    letter-spacing:.06em; color:var(--gray-3);
    margin-bottom:.35rem; font-weight:600;
}
.field input {
    width:100%; padding:.65rem .8rem;
    border:1px solid var(--gray-4); border-radius:6px;
    font-size:.9rem; font-family:inherit; color:var(--gray-1);
    background:var(--white); outline:none;
    transition:border-color .15s, box-shadow .15s;
}
.field input:focus {
    border-color:var(--yale-blue-lighter);
    box-shadow:0 0 0 3px rgba(40,109,192,.18);
}
.error {
    background:rgba(192,57,43,.08); color:var(--error);
    border:1px solid rgba(192,57,43,.25);
    padding:.55rem .75rem; border-radius:6px;
    font-size:.82rem; margin-bottom:1rem;
}
button.submit {
    width:100%; padding:.7rem 1rem;
    background:var(--yale-blue); color:var(--white);
    border:none; border-radius:6px;
    font-size:.9rem; font-weight:600; font-family:inherit;
    cursor:pointer;
    transition:background .15s;
}
button.submit:hover { background:var(--yale-blue-dark); }
.footer-note { text-align:center; margin-top:1.25rem; font-size:.7rem; color:var(--gray-3); }
</style>
</head>
<body>
<form class="login-card" method="POST" action="{{ url_for('login') }}{% if next %}?next={{ next }}{% endif %}">
    <div class="brand">
        <div class="brand-mark">Y</div>
        <h1>Yale Scanning QC</h1>
        <p>Sign in to continue</p>
    </div>
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    <input type="hidden" name="next" value="{{ next }}">
    <div class="field">
        <label for="u">Username</label>
        <input id="u" name="username" type="text" autocomplete="username" required autofocus>
    </div>
    <div class="field">
        <label for="p">Password</label>
        <input id="p" name="password" type="password" autocomplete="current-password" required>
    </div>
    <button class="submit" type="submit">Sign in</button>
    <div class="footer-note">Authorized personnel only</div>
</form>
</body>
</html>
'''


def main():
    """Run the Flask app."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Yale Scanning QC web app")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")

    args = parser.parse_args()

    print(f"\nStarting Yale Scanning QC (batch mode) at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
