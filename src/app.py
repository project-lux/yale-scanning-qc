"""
Flask web application for the Yale Scanning QC workflow (batch edition).

No real-time watching: a scheduler (or a manual "Run pass now") sweeps the
monitored directories, runs the active model versions over new scans, and
stores everything in a local SQLite database. Reviewers triage flagged
scans the next day; their verdicts become training data for the next
model version.
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


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


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


@app.route("/api/synthetic/badscan", methods=["POST"])
def api_gen_synthetic():
    """Manufacture synthetic degraded bad-scan examples from monitored images."""
    from . import synth
    dirs = db.list_directories(enabled_only=True)
    if not dirs:
        return jsonify({"error": "No monitored directories configured"}), 400
    imgs = []
    for d in dirs:
        imgs.extend(pipeline.discover_files(Path(d["path"])))
    if len(imgs) < 8:
        return jsonify({"error": "Not enough source images to synthesize from"}), 400
    data = request.get_json(force=True) if request.is_json else {}
    per_class = int(data.get("per_class") or 30)
    counts = synth.generate(imgs, per_class=per_class)
    return jsonify({"generated": counts})


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


# ---------------------------------------------------------------------------
# Main HTML Template — Yale Scanning QC (batch edition)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Yale Scanning QC</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --yale-blue: #00356b;
    --yale-blue-light: #286dc0;
    --yale-blue-lighter: #63aaff;
    --gray-1: #222222;
    --gray-2: #4a4a4a;
    --gray-3: #978d85;
    --gray-4: #dddddd;
    --gray-5: #f9f9f9;
    --white: #ffffff;
    --green: #5f712d;
    --green-bg: #eef3e5;
    --orange: #bd5319;
    --orange-bg: #fdf0ea;
    --red: #e04040;
    --sidebar-w: 320px;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Inter',system-ui,sans-serif; background:var(--gray-5); color:var(--gray-1); min-height:100vh; }
.app { display:flex; min-height:100vh; }

/* ---- Sidebar ---- */
.sidebar {
    width:var(--sidebar-w); background:var(--yale-blue); color:var(--white);
    display:flex; flex-direction:column; position:fixed; top:0; left:0; height:100vh; z-index:10;
}
.sidebar-header { padding:1.25rem 1.25rem 1rem; border-bottom:1px solid rgba(255,255,255,.15); }
.sidebar-header h1 { font-size:1.15rem; font-weight:700; letter-spacing:.02em; }
.sidebar-header p { font-size:.75rem; opacity:.65; margin-top:.2rem; }
.sidebar-header-row { display:flex; align-items:flex-start; justify-content:space-between; gap:.75rem; }
.logout-link {
    color:rgba(255,255,255,.7); font-size:.7rem; text-decoration:none;
    padding:.3rem .55rem; border:1px solid rgba(255,255,255,.2);
    border-radius:5px; transition:all .15s; flex-shrink:0;
}
.logout-link:hover { background:rgba(255,255,255,.1); color:var(--white); border-color:rgba(255,255,255,.4); }

/* ---- Nav ---- */
.nav { padding:.75rem; border-bottom:1px solid rgba(255,255,255,.15); display:flex; flex-direction:column; gap:.25rem; }
.nav-item {
    display:flex; align-items:center; gap:.6rem; padding:.5rem .75rem;
    border-radius:6px; font-size:.85rem; font-weight:500; cursor:pointer;
    color:rgba(255,255,255,.8); transition:all .12s; user-select:none;
    border-left:3px solid transparent;
}
.nav-item:hover { background:rgba(255,255,255,.08); color:var(--white); }
.nav-item.active { background:rgba(255,255,255,.12); color:var(--white); border-left-color:var(--yale-blue-lighter); }
.nav-item svg { width:16px; height:16px; flex-shrink:0; opacity:.8; }
.nav-badge {
    margin-left:auto; font-size:.62rem; font-weight:700; padding:.1rem .4rem;
    border-radius:8px; background:var(--orange); color:var(--white);
}

/* ---- Run status strip ---- */
.run-strip {
    padding:.6rem 1.25rem; border-bottom:1px solid rgba(255,255,255,.15);
    font-size:.72rem; color:rgba(255,255,255,.75); line-height:1.6;
}
.run-strip .run-label { text-transform:uppercase; letter-spacing:.07em; font-size:.62rem; opacity:.6; display:block; }
.run-strip .prog-bar { height:4px; background:rgba(255,255,255,.15); border-radius:2px; margin-top:.35rem; overflow:hidden; }
.run-strip .prog-fill { height:100%; background:var(--yale-blue-lighter); border-radius:2px; transition:width .3s; }
.run-btn {
    margin-top:.5rem; width:100%; padding:.45rem; border:1px solid rgba(255,255,255,.25);
    border-radius:6px; background:rgba(255,255,255,.08); color:var(--white);
    font-size:.75rem; font-weight:600; cursor:pointer; font-family:inherit; transition:background .15s;
}
.run-btn:hover { background:rgba(255,255,255,.18); }
.run-btn:disabled { opacity:.4; cursor:not-allowed; }
.run-btn.cancel { border-color:rgba(224,64,64,.6); background:rgba(224,64,64,.18); }
.run-btn.cancel:hover { background:rgba(224,64,64,.32); }

/* ---- New-pass modal ---- */
.runmodal-section { margin-bottom:1rem; }
.runmodal-section h3 {
    font-size:.68rem; text-transform:uppercase; letter-spacing:.07em;
    color:var(--gray-3); font-weight:600; margin-bottom:.5rem;
}
.runmodal-dir {
    display:flex; align-items:center; gap:.6rem; padding:.45rem .6rem;
    border:1px solid var(--gray-4); border-radius:6px; margin-bottom:.4rem;
    font-size:.78rem;
}
.runmodal-dir input { accent-color:var(--yale-blue); width:15px; height:15px; flex-shrink:0; }
.runmodal-dir .path { flex:1; word-break:break-all; font-family:'SF Mono','Fira Code',monospace; }
.runmodal-dir .counts { flex-shrink:0; font-size:.7rem; color:var(--gray-3); white-space:nowrap; }
.runmodal-dir .counts b { color:var(--yale-blue-light); }
.runmodal-select { width:100%; padding:.5rem .65rem; border:1px solid var(--gray-4); border-radius:6px;
    font-size:.83rem; font-family:inherit; color:var(--gray-1); background:var(--white); outline:none; }
.runmodal-summary { font-size:.8rem; color:var(--gray-2); padding:.55rem .7rem;
    background:var(--gray-5); border-radius:6px; margin-bottom:1rem; }
.runmodal-summary b { color:var(--yale-blue); }

/* ---- Batch picker + filters ---- */
.batch-picker { padding:.75rem 1.25rem; border-bottom:1px solid rgba(255,255,255,.15); }
.batch-picker label { font-size:.62rem; text-transform:uppercase; letter-spacing:.08em; opacity:.6; display:block; margin-bottom:.35rem; }
.batch-picker select {
    width:100%; padding:.45rem .55rem; border:1px solid rgba(255,255,255,.25); border-radius:6px;
    background:var(--yale-blue); color:var(--white); font-size:.8rem; font-family:inherit; outline:none;
}
.filter-chips { display:flex; gap:.3rem; margin-top:.55rem; flex-wrap:wrap; }
.chip {
    font-size:.66rem; font-weight:600; padding:.22rem .55rem; border-radius:10px;
    background:rgba(255,255,255,.1); color:rgba(255,255,255,.75); cursor:pointer;
    border:1px solid transparent; transition:all .12s; user-select:none;
}
.chip:hover { background:rgba(255,255,255,.2); }
.chip.active { background:var(--yale-blue-lighter); color:var(--yale-blue); }

/* ---- Scan list ---- */
.file-list-header {
    padding:.6rem 1.25rem; font-size:.7rem; text-transform:uppercase;
    letter-spacing:.08em; opacity:.6; border-bottom:1px solid rgba(255,255,255,.1);
}
.file-list { flex:1; overflow-y:auto; }
.tree-folder-header {
    display:flex; align-items:center; gap:.4rem; padding:.42rem 1rem .42rem .8rem;
    cursor:pointer; font-size:.8rem; user-select:none; transition:background .1s;
}
.tree-folder-header:hover { background:rgba(255,255,255,.07); }
.tree-chevron { width:12px; text-align:center; font-size:.6rem; transition:transform .15s; flex-shrink:0; opacity:.7; }
.tree-chevron.collapsed { transform:rotate(-90deg); }
.tree-folder-icon { flex-shrink:0; opacity:.7; }
.tree-folder-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:500; min-width:0; }
.tree-folder-count { font-size:.62rem; opacity:.5; margin-left:auto; white-space:nowrap; flex-shrink:0; }
.tree-file {
    display:flex; align-items:center; gap:.5rem; padding:.38rem 1rem .38rem 1.1rem;
    cursor:pointer; font-size:.78rem; border-left:3px solid transparent; transition:background .1s;
}
.tree-file:hover { background:rgba(255,255,255,.07); }
.tree-file.active { background:rgba(255,255,255,.12); border-left-color:var(--yale-blue-lighter); }
.file-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.dot-bad { background:var(--orange); box-shadow:0 0 6px var(--orange); }
.dot-clean { background:#8cc63f; }
.dot-unknown { background:#8899bb; }
.dot-error { background:var(--red); }
.tree-file-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.file-badge {
    font-size:.58rem; text-transform:uppercase; letter-spacing:.04em; padding:.12rem .35rem;
    border-radius:4px; font-weight:600; flex-shrink:0;
}
.badge-bad { background:var(--orange); color:var(--white); }
.badge-clean { background:rgba(140,198,63,.25); color:#8cc63f; }
.badge-unknown { background:rgba(255,255,255,.12); color:rgba(255,255,255,.6); }
.badge-error { background:var(--red); color:var(--white); }
.badge-reviewed { background:rgba(99,170,255,.2); color:var(--yale-blue-lighter); }
.empty-side { padding:1.2rem; font-size:.78rem; opacity:.55; line-height:1.6; }

/* ---- Main ---- */
.main { flex:1; margin-left:var(--sidebar-w); display:flex; flex-direction:column; min-height:100vh; }
.topbar {
    background:var(--white); border-bottom:1px solid var(--gray-4); padding:.85rem 1.5rem;
    display:flex; align-items:center; gap:.75rem; min-height:56px; flex-wrap:nowrap;
}
.topbar-title {
    font-size:.88rem; line-height:1.5; flex:1 1 auto; min-width:0;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.topbar-dir { color:var(--gray-3); font-weight:400; font-size:.78rem; }
.topbar-name { font-weight:600; }
.topbar-status { font-size:.78rem; font-weight:600; padding:.25rem .7rem; border-radius:5px; flex-shrink:0; }
.status-pass { background:var(--green-bg); color:var(--green); }
.status-fail { background:var(--orange-bg); color:var(--orange); }
.status-pending { background:#eee; color:var(--gray-3); }
.topbar-actions { margin-left:auto; display:flex; gap:.5rem; flex-shrink:0; flex-wrap:wrap; }
.btn {
    padding:.5rem 1.1rem; border:none; border-radius:6px; font-size:.82rem;
    font-weight:600; cursor:pointer; font-family:inherit; transition:all .15s;
}
.btn-accept { background:var(--orange); color:var(--white); }
.btn-accept:hover { background:#d4621f; }
.btn-ignore { background:var(--gray-4); color:var(--gray-2); }
.btn-ignore:hover { background:#ccc; }
.btn-flag { background:var(--red); color:var(--white); }
.btn-flag:hover { background:#c93030; }
.btn-good { background:var(--green); color:var(--white); }
.btn-good:hover { background:#4e5d25; }
.btn-blue { background:var(--yale-blue); color:var(--white); }
.btn-blue:hover { background:var(--yale-blue-light); }
.btn:disabled { opacity:.45; cursor:not-allowed; }
.topbar-nav { display:flex; gap:.4rem; flex-shrink:0; margin-left:.5rem; order:99; }
.btn-nav {
    background:var(--gray-4); color:var(--gray-2); padding:.5rem 0; font-size:.85rem;
    line-height:1; width:4.6rem; text-align:center; flex-shrink:0;
}
.btn-nav:hover { background:#ccc; }
.tree-folder-header.dirv-side-sel { background:rgba(255,255,255,.12); }

.content { flex:1; padding:1.5rem 2rem; overflow-y:auto; }
.empty-state {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    min-height:400px; color:var(--gray-3); gap:.75rem; text-align:center;
}
.empty-state svg { width:64px; height:64px; opacity:.35; }
.empty-state p { font-size:1rem; }
.empty-state .hint { font-size:.82rem; opacity:.7; }

/* ---- Left-pane tabs (Predictions / Transcription) ---- */
.tab-bar { display:flex; gap:.25rem; border-bottom:1px solid var(--gray-4); margin-bottom:1rem; }
.tab {
    padding:.5rem .9rem; font-size:.8rem; font-weight:600; color:var(--gray-3);
    cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-1px;
    display:flex; align-items:center; gap:.45rem; user-select:none; transition:color .12s;
}
.tab:hover { color:var(--gray-1); }
.tab.active { color:var(--yale-blue); border-bottom-color:var(--yale-blue); }
.tab-ind { width:7px; height:7px; border-radius:50%; background:var(--gray-4); flex-shrink:0; }
.tab-ind.run { background:var(--green); }
.tab-ind.partial { background:var(--orange); }

/* ---- Per-stage rerun buttons ---- */
.stage-rerun-row { display:flex; gap:.5rem; margin-bottom:1.25rem; flex-wrap:wrap; }

/* ---- Stage indicators in the scan list (M/P/T) ---- */
.stage-inds { display:flex; gap:2px; flex-shrink:0; }
.stage-ind {
    font-size:.52rem; font-weight:700; width:13px; height:13px; border-radius:3px;
    display:flex; align-items:center; justify-content:center;
    background:rgba(255,255,255,.1); color:rgba(255,255,255,.3);
}
.stage-ind.run { background:rgba(140,198,63,.3); color:#c9e9a2; }

/* ---- Scan review layout: outputs left, full image right ---- */
.scan-layout { display:grid; grid-template-columns:minmax(320px,34%) 1fr; gap:1.25rem; align-items:start; }
.scan-layout .checks-grid { flex-direction:column; }
.scan-layout .check-card { min-width:0; flex:none; }
.scan-layout .review-card { flex-direction:column; align-items:stretch; gap:.9rem; }
.scan-layout .review-field select { min-width:0; width:100%; }
.scan-layout .image-card { position:sticky; top:1rem; }
.scan-layout .preview-img { max-height:calc(100vh - 190px); width:auto; max-width:100%; margin:0 auto; }
.scan-layout .image-container { display:flex; justify-content:center; }
@media(max-width:1100px){
    .scan-layout { grid-template-columns:1fr; }
    .scan-layout .image-card { position:static; }
}

/* ---- Cards ---- */
.checks-grid { display:flex; gap:.75rem; flex-wrap:wrap; margin-bottom:1.25rem; }
.check-card {
    background:var(--white); border:1px solid var(--gray-4); border-radius:10px;
    padding:1rem 1.15rem; min-width:200px; flex:1;
    transition:box-shadow .15s, border-color .15s;
}
.check-card:hover { border-color:var(--yale-blue-lighter); box-shadow:0 2px 8px rgba(0,53,107,.1); }
.check-header { display:flex; align-items:center; gap:.5rem; margin-bottom:.5rem; }
.check-dot { width:10px; height:10px; border-radius:50%; }
.check-dot.pass { background:var(--green); }
.check-dot.fail { background:var(--orange); }
.check-dot.unknown { background:var(--gray-3); }
.check-name { font-size:.85rem; font-weight:600; }
.check-detail { font-size:.75rem; color:var(--gray-2); line-height:1.5; }
.check-conf { margin-top:.5rem; height:4px; background:var(--gray-4); border-radius:2px; overflow:hidden; }
.check-conf-fill { height:100%; border-radius:2px; }
.conf-pass { background:var(--green); }
.conf-fail { background:var(--orange); }
.conf-unknown { background:var(--gray-3); }

/* ---- Review panel ---- */
.review-card {
    background:var(--white); border:1px solid var(--gray-4); border-radius:10px;
    padding:1rem 1.15rem; margin-bottom:1.25rem;
    display:flex; gap:1.25rem; align-items:flex-end; flex-wrap:wrap;
}
.review-field { display:flex; flex-direction:column; gap:.35rem; }
.review-field label {
    font-size:.68rem; text-transform:uppercase; letter-spacing:.06em;
    color:var(--gray-3); font-weight:600;
}
.review-field select, .review-field input[type=text] {
    padding:.5rem .65rem; border:1px solid var(--gray-4); border-radius:6px;
    font-size:.83rem; font-family:inherit; color:var(--gray-1); background:var(--white); outline:none;
    min-width:180px;
}
.review-field select:focus, .review-field input:focus { border-color:var(--yale-blue-lighter); }
.review-note { flex:1; min-width:220px; }
.review-note input { width:100%; }
.review-saved { font-size:.75rem; color:var(--green); font-weight:600; }

/* ---- Transcription ---- */
.transcript-card {
    background:var(--white); border:1px solid var(--gray-4); border-radius:10px;
    padding:1rem 1.15rem; margin-bottom:1.25rem;
}
.transcript-card h3 {
    font-size:.72rem; text-transform:uppercase; letter-spacing:.08em;
    color:var(--gray-3); margin-bottom:.6rem; display:flex; gap:.5rem; align-items:center;
}
.transcript-backend {
    font-size:.6rem; background:var(--gray-5); border:1px solid var(--gray-4);
    padding:.1rem .4rem; border-radius:4px; color:var(--gray-2); text-transform:none; letter-spacing:0;
}
.transcript-text {
    font-size:.85rem; line-height:1.7; white-space:pre-wrap; color:var(--gray-1);
    height:440px; min-height:140px; resize:vertical; overflow:auto;
    font-family:'SF Mono','Fira Code',Georgia,serif;
    border:1px solid var(--gray-5); border-radius:6px; padding:.6rem .75rem;
}

/* ---- Directories view: real folder tree ---- */
.dirv-header { display:flex; align-items:center; gap:.75rem; flex-wrap:wrap; }
.dirv-path { font-family:'SF Mono','Fira Code',monospace; font-size:.82rem; word-break:break-all; }
.dirv-counts { font-size:.72rem; color:var(--gray-3); white-space:nowrap; font-weight:400; }
.dirv-counts b { color:var(--yale-blue-light); }
.dirv-tree { border:1px solid var(--gray-4); border-radius:8px; overflow:hidden; }
.dirv-row {
    display:flex; align-items:center; gap:.5rem; padding:.34rem .75rem;
    font-size:.8rem; border-top:1px solid var(--gray-5); min-height:32px;
}
.dirv-tree > .dirv-row:first-child { border-top:none; }
.dirv-row:hover { background:var(--gray-5); }
.dirv-row.folder { cursor:pointer; user-select:none; font-weight:600; }
.dirv-row .chev {
    font-size:.58rem; width:12px; text-align:center; flex-shrink:0;
    color:var(--gray-3); transition:transform .15s;
}
.dirv-row .chev.open { transform:rotate(90deg); }
.dirv-row .ficon { width:15px; height:15px; flex-shrink:0; color:var(--yale-blue-light); }
.dirv-row .grow { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
.dirv-row .rowbtns { display:flex; gap:.35rem; flex-shrink:0; }
.dirv-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.dirv-dot.processed { background:#8cc63f; }
.dirv-dot.new { background:var(--gray-4); }
.dirv-dot.error { background:var(--red); }
.dirv-row .stage-ind { background:var(--gray-4); color:var(--gray-3); }
.dirv-row .stage-ind.run { background:var(--green-bg); color:var(--green); }

/* ---- Image viewer ---- */
.image-card { background:var(--white); border:1px solid var(--gray-4); border-radius:10px; padding:1.25rem; }
.image-card-header {
    display:flex; align-items:center; justify-content:space-between; margin-bottom:.75rem;
    flex-wrap:wrap; gap:.5rem;
}
.image-card-header h3 {
    font-size:.72rem; text-transform:uppercase; letter-spacing:.08em;
    color:var(--gray-3); margin:0;
}
.viewer-toolbar { display:flex; align-items:center; gap:.75rem; font-size:.78rem; color:var(--gray-2); }
.zoom-controls { display:flex; align-items:center; gap:.25rem; }
.zoom-controls button {
    width:26px; height:26px; border:1px solid var(--gray-4); border-radius:4px;
    background:var(--white); cursor:pointer; font-size:.85rem; display:flex;
    align-items:center; justify-content:center; color:var(--gray-2); transition:all .1s;
}
.zoom-controls button:hover { background:var(--gray-5); border-color:var(--gray-3); }
.zoom-level { min-width:38px; text-align:center; font-size:.72rem; color:var(--gray-3); font-variant-numeric:tabular-nums; }
.image-container { overflow:hidden; border-radius:6px; background:var(--gray-5); cursor:grab; position:relative; }
.image-container.panning { cursor:grabbing; }
.image-transform { transform-origin:0 0; }
.preview-img { display:block; max-width:100%; max-height:70vh; border-radius:6px; }

/* ---- Per-image pass versions ---- */
.versions-card { background:var(--white); border:1px solid var(--gray-4); border-radius:10px; padding:.9rem 1.15rem; margin-bottom:1.25rem; }
.versions-card h3 { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:var(--gray-3); margin-bottom:.6rem; }
.version-row {
    display:flex; align-items:center; gap:.6rem; padding:.4rem .5rem; border-radius:6px;
    font-size:.78rem; cursor:pointer; border:1px solid transparent;
}
.version-row:hover { background:var(--gray-5); }
.version-row.active { border-color:var(--yale-blue-lighter); background:#f4f8ff; }
.version-row .vpill { font-size:.6rem; font-weight:700; padding:.12rem .4rem; border-radius:6px; text-transform:uppercase; }
.vpill.bad { background:var(--orange-bg); color:var(--orange); }
.vpill.clean { background:var(--green-bg); color:var(--green); }
.vpill.unk { background:var(--gray-5); color:var(--gray-3); border:1px solid var(--gray-4); }
.version-meta { color:var(--gray-3); font-size:.72rem; }

/* ---- Dataset browser ---- */
.ds-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(120px,1fr)); gap:.6rem; }
.ds-thumb { border:1px solid var(--gray-4); border-radius:8px; overflow:hidden; background:var(--gray-5); }
.ds-thumb img { width:100%; height:110px; object-fit:cover; display:block; }
.ds-thumb .cap { font-size:.62rem; padding:.25rem .4rem; display:flex; justify-content:space-between; gap:.3rem; }
.ds-thumb .cap .lbl { font-weight:700; }
.ds-thumb .cap .src { color:var(--gray-3); }
.ds-classbar { display:flex; gap:.4rem; flex-wrap:wrap; margin-bottom:.8rem; }
.ds-classchip { font-size:.7rem; padding:.2rem .55rem; border-radius:10px; background:var(--gray-5); border:1px solid var(--gray-4); cursor:pointer; }
.ds-classchip.active { background:var(--yale-blue); color:var(--white); border-color:var(--yale-blue); }

/* ---- Tables (batches / models) ---- */
.panel {
    background:var(--white); border:1px solid var(--gray-4); border-radius:10px;
    padding:1.25rem; margin-bottom:1.5rem;
}
.panel h2 { font-size:.95rem; font-weight:600; margin-bottom:.9rem; display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }
.panel h2 .spacer { margin-left:auto; }
.panel-sub { font-size:.78rem; color:var(--gray-3); margin:-.5rem 0 .9rem; line-height:1.5; }
table.data { width:100%; border-collapse:collapse; font-size:.8rem; }
table.data th {
    text-align:left; font-size:.66rem; text-transform:uppercase; letter-spacing:.06em;
    color:var(--gray-3); font-weight:600; padding:.45rem .6rem; border-bottom:1px solid var(--gray-4);
}
table.data td { padding:.5rem .6rem; border-bottom:1px solid var(--gray-5); vertical-align:middle; }
table.data tr:hover td { background:var(--gray-5); }
.pill {
    font-size:.62rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em;
    padding:.15rem .45rem; border-radius:8px; display:inline-block;
}
.pill-done { background:var(--green-bg); color:var(--green); }
.pill-running { background:#e3edff; color:var(--yale-blue-light); }
.pill-error { background:#fde5e5; color:var(--red); }
.pill-active { background:var(--yale-blue); color:var(--white); }
.pill-neutral { background:var(--gray-5); color:var(--gray-2); border:1px solid var(--gray-4); }
.mini-btn {
    padding:.3rem .7rem; border:1px solid var(--gray-4); border-radius:5px;
    background:var(--white); font-size:.72rem; font-weight:600; cursor:pointer;
    font-family:inherit; color:var(--gray-2); transition:all .12s;
}
.mini-btn:hover { border-color:var(--yale-blue-light); color:var(--yale-blue); }
.mini-btn.primary { background:var(--yale-blue); color:var(--white); border-color:var(--yale-blue); }
.mini-btn.primary:hover { background:var(--yale-blue-light); }
.mini-btn.danger:hover { border-color:var(--red); color:var(--red); }
.mini-btn:disabled { opacity:.45; cursor:not-allowed; }

/* ---- Settings ---- */
.set-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:1.5rem; align-items:start; }
.set-field { margin-bottom:.9rem; }
.set-field label {
    display:block; font-size:.7rem; text-transform:uppercase; letter-spacing:.06em;
    color:var(--gray-3); margin-bottom:.35rem; font-weight:600;
}
.set-field input[type=text], .set-field input[type=time], .set-field input[type=number], .set-field select {
    width:100%; padding:.55rem .7rem; border:1px solid var(--gray-4); border-radius:6px;
    font-size:.85rem; font-family:inherit; color:var(--gray-1); background:var(--white); outline:none;
}
.set-field input:focus, .set-field select:focus { border-color:var(--yale-blue-lighter); }
.set-row { display:flex; gap:.6rem; align-items:center; }
.set-check { display:flex; align-items:center; gap:.5rem; font-size:.85rem; cursor:pointer; user-select:none; }
.set-check input { accent-color:var(--yale-blue); width:16px; height:16px; }
.dir-item {
    display:flex; align-items:center; gap:.6rem; padding:.5rem .65rem;
    border:1px solid var(--gray-4); border-radius:6px; margin-bottom:.45rem;
    font-size:.8rem; font-family:'SF Mono','Fira Code',monospace;
}
.dir-item .grow { flex:1; word-break:break-all; }
.dir-item.disabled { opacity:.5; }
.save-note { font-size:.75rem; color:var(--green); font-weight:600; margin-left:.6rem; }

/* ---- Modal (folder browser) ---- */
.modal-backdrop {
    display:none; position:fixed; inset:0; background:rgba(0,0,0,.4);
    z-index:200; align-items:center; justify-content:center;
}
.modal-backdrop.active { display:flex; }
.modal {
    background:var(--white); border-radius:12px; width:520px; max-width:92vw;
    max-height:80vh; display:flex; flex-direction:column;
    box-shadow:0 12px 40px rgba(0,0,0,.2);
}
.modal-header {
    padding:1rem 1.25rem; border-bottom:1px solid var(--gray-4);
    display:flex; align-items:center; justify-content:space-between;
}
.modal-header h2 { font-size:.95rem; font-weight:600; color:var(--gray-1); }
.modal-close {
    background:none; border:none; font-size:1.2rem; cursor:pointer;
    color:var(--gray-3); padding:.2rem .4rem; border-radius:4px;
}
.modal-close:hover { color:var(--gray-1); background:var(--gray-5); }
.modal-path {
    padding:.6rem 1.25rem; background:var(--gray-5); border-bottom:1px solid var(--gray-4);
    font-size:.78rem; color:var(--gray-2); font-family:monospace;
    display:flex; align-items:center; gap:.5rem; word-break:break-all;
}
.modal-path .tiff-hint {
    margin-left:auto; font-family:'Inter',sans-serif; font-size:.7rem;
    color:var(--yale-blue-light); font-weight:600; flex-shrink:0;
}
.modal-body { flex:1; overflow-y:auto; padding:.5rem 0; }
.browse-item {
    display:flex; align-items:center; gap:.6rem; padding:.5rem 1.25rem;
    cursor:pointer; font-size:.85rem; color:var(--gray-1); transition:background .1s;
}
.browse-item:hover { background:var(--gray-5); }
.browse-item svg { width:18px; height:18px; color:var(--yale-blue-light); flex-shrink:0; }
.browse-up { color:var(--gray-3); font-style:italic; }
.browse-empty { padding:1.5rem 1.25rem; text-align:center; color:var(--gray-3); font-size:.85rem; }
.modal-footer {
    padding:.75rem 1.25rem; border-top:1px solid var(--gray-4);
    display:flex; justify-content:flex-end; gap:.5rem;
}
.modal-footer .btn-select {
    padding:.5rem 1.2rem; background:var(--yale-blue); color:var(--white);
    border:none; border-radius:6px; font-size:.82rem; font-weight:600; cursor:pointer; font-family:inherit;
}
.modal-footer .btn-select:hover { background:var(--yale-blue-light); }
.modal-footer .btn-cancel {
    padding:.5rem 1.2rem; background:var(--gray-4); color:var(--gray-2);
    border:none; border-radius:6px; font-size:.82rem; font-weight:600; cursor:pointer; font-family:inherit;
}
.modal-footer .btn-cancel:hover { background:#ccc; }

.loading-overlay {
    display:none; position:fixed; inset:0; background:rgba(0,0,0,.25);
    z-index:100; align-items:center; justify-content:center;
}
.loading-overlay.active { display:flex; }
.spinner-box {
    background:var(--white); padding:2rem 2.5rem; border-radius:12px;
    box-shadow:0 8px 30px rgba(0,0,0,.15); text-align:center;
}
.spinner {
    width:36px; height:36px; border:3px solid var(--gray-4);
    border-top-color:var(--yale-blue); border-radius:50%;
    animation:spin .8s linear infinite; margin:0 auto .75rem;
}
@keyframes spin { to{transform:rotate(360deg)} }
.spinner-box p { font-size:.88rem; color:var(--gray-2); }

@media(max-width:900px){
    .topbar { padding:.75rem 1rem; }
    .content { padding:1rem; }
    .checks-grid { flex-direction:column; }
}
</style>
</head>
<body>
<div class="app">

<aside class="sidebar">
    <div class="sidebar-header">
        <div class="sidebar-header-row">
            <div>
                <h1>Yale Scanning QC</h1>
                <p>Batch Quality Control</p>
            </div>
            <a class="logout-link" href="/logout" title="Sign out">Sign out</a>
        </div>
    </div>

    <nav class="nav">
        <div class="nav-item" data-view="review">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
            Review <span class="nav-badge" id="reviewBadge" style="display:none"></span>
        </div>
        <div class="nav-item" data-view="directories">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>
            Directories
        </div>
        <div class="nav-item" data-view="badscans">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            Bad Scans <span class="nav-badge" id="badBadge" style="display:none"></span>
        </div>
        <div class="nav-item" data-view="datasets">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0018 0V5"/><path d="M3 12a9 3 0 0018 0"/></svg>
            Datasets
        </div>
        <div class="nav-item" data-view="batches">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="4" rx="1"/><rect x="3" y="10" width="18" height="4" rx="1"/><rect x="3" y="16" width="18" height="4" rx="1"/></svg>
            Batches
        </div>
        <div class="nav-item" data-view="models">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l9 4.9v10.2L12 22l-9-4.9V6.9L12 2z"/><path d="M12 22V12"/><path d="M21 6.9L12 12 3 6.9"/></svg>
            Models
        </div>
        <div class="nav-item" data-view="settings">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 11-4 0v-.09a1.65 1.65 0 00-1-1.51 1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 110-4h.09a1.65 1.65 0 001.51-1 1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33h0a1.65 1.65 0 001-1.51V3a2 2 0 114 0v.09a1.65 1.65 0 001 1.51h0a1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82v0a1.65 1.65 0 001.51 1H21a2 2 0 110 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
            Settings
        </div>
    </nav>

    <div class="run-strip" id="runStrip">
        <span class="run-label">Pipeline</span>
        <span id="runStatusText">Idle</span>
        <div class="prog-bar" id="runProgBar" style="display:none"><div class="prog-fill" id="runProgFill" style="width:0%"></div></div>
        <button class="run-btn" id="runNowBtn">New pass&hellip;</button>
    </div>

    <div class="batch-picker" id="batchPicker" style="display:none">
        <label>Batch</label>
        <select id="batchSelect"></select>
        <div class="filter-chips" id="filterChips">
            <span class="chip active" data-filter="flagged">Flagged</span>
            <span class="chip" data-filter="unreviewed">Unreviewed</span>
            <span class="chip" data-filter="all">All</span>
        </div>
    </div>

    <div class="file-list-header" id="fileListHeader" style="display:none">Scans</div>
    <div class="file-list" id="fileList"></div>
</aside>

<div class="main">
    <div class="topbar" id="topbar" style="display:none">
        <div class="topbar-title" id="topFilename"></div>
        <div class="topbar-status" id="topStatus"></div>
        <div class="topbar-actions" id="topActions"></div>
        <div class="topbar-nav" id="topNav"></div>
    </div>
    <div class="content" id="content"></div>
</div>
</div>

<div class="modal-backdrop" id="folderModal">
    <div class="modal">
        <div class="modal-header">
            <h2>Select Folder</h2>
            <button class="modal-close" id="modalClose">&times;</button>
        </div>
        <div class="modal-path">
            <span id="modalPath">/</span>
            <span class="tiff-hint" id="modalTiffHint"></span>
        </div>
        <div class="modal-body" id="modalBody"></div>
        <div class="modal-footer">
            <button class="btn-cancel" id="modalCancel">Cancel</button>
            <button class="btn-select" id="modalSelect">Add This Folder</button>
        </div>
    </div>
</div>

<div class="modal-backdrop" id="runModal">
    <div class="modal">
        <div class="modal-header">
            <h2>New QC Pass</h2>
            <button class="modal-close" id="runModalClose">&times;</button>
        </div>
        <div class="modal-body" id="runModalBody" style="padding:1rem 1.25rem"></div>
        <div class="modal-footer">
            <button class="btn-cancel" id="runModalCancel">Cancel</button>
            <button class="btn-select" id="runModalStart">Start pass</button>
        </div>
    </div>
</div>

<div class="modal-backdrop" id="rerunModal">
    <div class="modal">
        <div class="modal-header">
            <h2 id="rerunModalTitle">Re-run</h2>
            <button class="modal-close" id="rerunModalClose">&times;</button>
        </div>
        <div class="modal-body" style="padding:1rem 1.25rem">
            <div class="runmodal-section">
                <label class="set-check" style="margin-bottom:.5rem">
                    <input type="radio" name="rerunMode" value="full" checked>
                    <span><b>Full re-pass</b> — re-process all files as a new versioned batch (all stages, currently active models)</span>
                </label>
                <label class="set-check">
                    <input type="radio" name="rerunMode" value="stages">
                    <span><b>Update in place</b> — re-run selected stages on this batch&rsquo;s existing scans</span>
                </label>
            </div>
            <div class="runmodal-section" id="rerunStagesBox" style="opacity:.45;pointer-events:none">
                <h3>Stages</h3>
                <label class="set-check" style="margin-bottom:.35rem"><input type="checkbox" class="rerunStage" value="badscan" checked> Bad-scan model</label>
                <label class="set-check" style="margin-bottom:.35rem"><input type="checkbox" class="rerunStage" value="pagetype" checked> Page-type detection</label>
                <label class="set-check"><input type="checkbox" class="rerunStage" value="transcription"> Transcription (handwriting pages)</label>
            </div>
        </div>
        <div class="modal-footer">
            <button class="btn-cancel" id="rerunModalCancel">Cancel</button>
            <button class="btn-select" id="rerunModalGo">Run</button>
        </div>
    </div>
</div>

<div class="loading-overlay" id="loadingOverlay">
    <div class="spinner-box">
        <div class="spinner"></div>
        <p id="loadingText">Loading...</p>
    </div>
</div>

<script>
const $ = s => document.querySelector(s);
const content = $('#content');
const topbar = $('#topbar');
const topFilename = $('#topFilename');
const topStatus = $('#topStatus');
const topActions = $('#topActions');
const topNav = $('#topNav');
const loadingOv = $('#loadingOverlay');
const loadingTxt = $('#loadingText');

let state = null;            // /api/state payload
let currentView = 'review';
let batches = [];
let currentBatchId = null;
let scans = [];
let selectedScanId = null;
let scanFilter = 'flagged';
let wasRunning = false;
let currentTab = 'pred';   // left-pane tab on the scan view

const PAGE_LABELS = {blank_page:'Blank page', handwriting:'Handwriting', photograph:'Photograph', typed_text:'Typed text'};
const VERDICT_LABELS = {
    true_positive:'Confirmed bad scan', false_positive:'False positive',
    false_negative:'Missed bad scan', true_negative:'Confirmed clean'
};

function showLoading(msg){ loadingTxt.textContent = msg || 'Loading...'; loadingOv.classList.add('active'); }
function hideLoading(){ loadingOv.classList.remove('active'); }
function esc(s){ const d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; }
function fmtTime(iso){
    if(!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

async function api(path, opts){
    const r = await fetch(path, opts);
    const d = await r.json().catch(() => ({}));
    if(!r.ok){ throw new Error(d.error || ('Request failed: ' + r.status)); }
    return d;
}
async function apiPost(path, body){
    return api(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body || {})});
}

/* ================= Init + polling ================= */
(async function init(){
    document.querySelectorAll('.nav-item').forEach(el => {
        el.onclick = () => switchView(el.dataset.view);
    });
    $('#runNowBtn').onclick = openRunModal;
    document.querySelectorAll('#filterChips .chip').forEach(ch => {
        ch.onclick = () => {
            document.querySelectorAll('#filterChips .chip').forEach(c => c.classList.remove('active'));
            ch.classList.add('active');
            scanFilter = ch.dataset.filter;
            renderScanList();
        };
    });
    $('#batchSelect').onchange = () => loadBatch(parseInt($('#batchSelect').value, 10));

    /* Left / Right arrows step through images (Review list or Directory tree) */
    document.addEventListener('keydown', e => {
        const tag = (e.target.tagName || '').toLowerCase();
        if(tag === 'input' || tag === 'textarea' || tag === 'select') return;
        let nav = null;
        if((currentView === 'review' || currentView === 'badscans') && selectedScanId) nav = navigateScan;
        if(currentView === 'directories' && dirvSelected && dirvSelected.type === 'file') nav = dirvNavigate;
        if(!nav) return;
        if(e.key === 'ArrowLeft'){ e.preventDefault(); nav(-1); }
        if(e.key === 'ArrowRight'){ e.preventDefault(); nav(1); }
    });

    await refreshState();
    await loadBatches(true);
    switchView('review');
    setInterval(pollTick, 3000);
})();

async function refreshState(){
    try { state = await api('/api/state'); } catch(e){ return; }
    renderRunStrip();
}

async function pollTick(){
    const prevRunning = state && (state.batch_running || state.train_running);
    await refreshState();
    const nowRunning = state && (state.batch_running || state.train_running);
    if(prevRunning && !nowRunning){
        // A batch or training just finished — refresh whatever is on screen.
        await loadBatches(currentBatchId == null);
        if(currentView === 'batches') renderBatchesView();
        if(currentView === 'models') renderModelsView();
        if(currentView === 'directories') renderDirectoriesView();
        if(currentView === 'review' && currentBatchId) loadBatch(currentBatchId);
    } else if(nowRunning && currentView === 'batches'){
        renderBatchesView();
    }
}

function renderRunStrip(){
    const txt = $('#runStatusText');
    const bar = $('#runProgBar');
    const fill = $('#runProgFill');
    const btn = $('#runNowBtn');
    if(!state){ txt.textContent = '—'; return; }

    if(state.batch_running && state.batch_progress && state.batch_progress.total !== undefined){
        const p = state.batch_progress;
        const what = p.stage_rerun ? ('Re-running ' + p.stage_rerun) : 'Running pass';
        txt.textContent = what + ': ' + p.done + '/' + p.total + (p.current ? ' · ' + p.current : '');
        bar.style.display = 'block';
        fill.style.width = (p.total ? Math.round(100 * p.done / p.total) : 0) + '%';
        btn.disabled = false;
        btn.textContent = 'Cancel pass';
        btn.classList.add('cancel');
        btn.onclick = cancelRunningBatch;
    } else if(state.train_running){
        const tp = state.train_progress || {};
        const where = tp.where === 'remote' ? ' on GPU box' : tp.where === 'local' ? ' locally' : '';
        let msg = 'Training ' + (tp.version || tp.role || 'model') + where;
        if(tp.status === 'materializing') msg += ' · preparing dataset';
        else if(tp.epoch && tp.epochs) msg += ' · epoch ' + tp.epoch + '/' + tp.epochs;
        else if(tp.status) msg += ' · ' + tp.status;
        txt.textContent = msg + '…';
        bar.style.display = tp.epoch && tp.epochs ? 'block' : 'none';
        if(tp.epoch && tp.epochs) fill.style.width = Math.round(100 * tp.epoch / tp.epochs) + '%';
        btn.disabled = true;
        btn.textContent = 'New pass…';
        btn.classList.remove('cancel');
    } else {
        txt.textContent = 'Idle · next scheduled run: ' + (state.next_scheduled || 'disabled');
        bar.style.display = 'none';
        btn.disabled = false;
        btn.textContent = 'New pass…';
        btn.classList.remove('cancel');
        btn.onclick = openRunModal;
    }
    renderRemoteStrip();
}

function renderRemoteStrip(){
    let el = $('#remoteStrip');
    const r = state && state.remote;
    if(!r || !r.enabled){ if(el) el.style.display = 'none'; return; }
    if(!el){
        el = document.createElement('div');
        el.id = 'remoteStrip';
        el.style.cssText = 'font-size:.66rem;margin-top:.45rem;opacity:.8;display:flex;align-items:center;gap:.4rem';
        $('#runStrip').appendChild(el);
    }
    el.style.display = 'flex';
    const dot = r.reachable ? '#8cc63f' : '#e04040';
    const label = r.reachable
        ? 'GPU: ' + (r.gpu || 'connected') + (r.inference ? ' · inference on' : '')
        : 'GPU box offline';
    el.innerHTML = '<span style="width:7px;height:7px;border-radius:50%;background:' + dot + ';flex-shrink:0"></span>'
        + '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(label) + '</span>';
}

async function cancelRunningBatch(){
    if(!confirm('Cancel the running pass? Files already processed stay recorded; the rest will be picked up by the next pass.')) return;
    try {
        await apiPost('/api/batches/cancel');
        await refreshState();
    } catch(e){ alert(e.message); }
}

/* ---- New-pass modal: scan first, then pick models ---- */
const runModal = $('#runModal');
$('#runModalClose').onclick = () => runModal.classList.remove('active');
$('#runModalCancel').onclick = () => runModal.classList.remove('active');
$('#runModalStart').onclick = startConfiguredPass;

async function openRunModal(){
    const body = $('#runModalBody');
    runModal.classList.add('active');
    body.innerHTML = '<div class="browse-empty">Scanning directories…</div>';
    $('#runModalStart').disabled = true;

    let preview, modelsData;
    try {
        [preview, modelsData] = await Promise.all([
            apiPost('/api/batches/preview'),
            api('/api/models'),
        ]);
    } catch(e){
        body.innerHTML = '<div class="browse-empty">' + esc(e.message) + '</div>';
        return;
    }

    let html = '<div class="runmodal-summary">Found <b>' + preview.new + '</b> new image'
        + (preview.new === 1 ? '' : 's') + ' to process (' + preview.total + ' total on disk).</div>';

    html += '<div class="runmodal-section"><h3>Directories</h3>';
    preview.directories.forEach((d, i) => {
        html += '<label class="runmodal-dir">'
            + '<input type="checkbox" class="runDirCheck" data-path="' + esc(d.path) + '"' + (d.new > 0 ? ' checked' : '') + '>'
            + '<span class="path">' + esc(d.path) + '</span>'
            + '<span class="counts"><b>' + d.new + '</b> new / ' + d.total + '</span>'
            + '</label>';
    });
    html += '</div>';

    const roleTitles = {badscan: 'Bad-scan model', pagetype: 'Page-type model'};
    ['badscan', 'pagetype'].forEach(role => {
        const models = (modelsData.models || []).filter(m => m.role === role);
        html += '<div class="runmodal-section"><h3>' + roleTitles[role] + '</h3>'
            + '<select class="runmodal-select" id="runModel_' + role + '">';
        models.forEach(m => {
            let acc = '';
            try {
                const met = JSON.parse(m.metrics || '{}');
                if(met.top1_accuracy) acc = ' — ' + (met.top1_accuracy * 100).toFixed(1) + '% acc';
            } catch(e){}
            html += '<option value="' + m.id + '"' + (m.is_active ? ' selected' : '') + '>'
                + esc(m.version) + (m.is_active ? ' (active)' : '') + acc + '</option>';
        });
        html += '</select></div>';
    });

    body.innerHTML = html;
    $('#runModalStart').disabled = false;

    body.querySelectorAll('.runDirCheck').forEach(cb => { cb.onchange = updateRunModalStart; });
    updateRunModalStart();
}

function updateRunModalStart(){
    const anyChecked = Array.from(document.querySelectorAll('.runDirCheck')).some(cb => cb.checked);
    $('#runModalStart').disabled = !anyChecked;
}

async function startConfiguredPass(){
    const dirs = Array.from(document.querySelectorAll('.runDirCheck'))
        .filter(cb => cb.checked).map(cb => cb.dataset.path);
    const payload = {directories: dirs};
    const mSel = document.getElementById('runModel_badscan');
    const pSel = document.getElementById('runModel_pagetype');
    if(mSel && mSel.value) payload.badscan_model_id = parseInt(mSel.value, 10);
    if(pSel && pSel.value) payload.pagetype_model_id = parseInt(pSel.value, 10);

    try {
        await apiPost('/api/batches/run', payload);
        runModal.classList.remove('active');
        await refreshState();
        if(currentView === 'batches') renderBatchesView();
    } catch(e){ alert(e.message); }
}

/* ================= View switching ================= */
function switchView(view){
    currentView = view;
    document.querySelectorAll('.nav-item').forEach(el =>
        el.classList.toggle('active', el.dataset.view === view));
    const isReview = view === 'review';
    const isDirs = view === 'directories';
    const isBad = view === 'badscans';
    $('#batchPicker').style.display = isReview ? 'block' : 'none';
    $('#fileListHeader').style.display = (isReview || isDirs || isBad) ? 'block' : 'none';
    $('#fileList').style.display = (isReview || isDirs || isBad) ? 'block' : 'none';
    if(isReview){ $('#fileListHeader').textContent = 'Scans'; renderScanList(); }
    topbar.style.display = 'none';

    if(view === 'review') renderReviewView();
    if(view === 'directories') renderDirectoriesView();
    if(view === 'badscans') renderBadScansView();
    if(view === 'datasets') renderDatasetsView();
    if(view === 'batches') renderBatchesView();
    if(view === 'models') renderModelsView();
    if(view === 'settings') renderSettingsView();
}

/* ================= Batches data ================= */
async function loadBatches(selectLatest){
    try {
        const d = await api('/api/batches');
        batches = d.batches || [];
    } catch(e){ batches = []; }

    const sel = $('#batchSelect');
    sel.innerHTML = '';
    batches.forEach(b => {
        const opt = document.createElement('option');
        opt.value = b.id;
        opt.textContent = b.label + ' · ' + fmtTime(b.started_at) + ' (' + (b.n_bad||0) + ' flagged)';
        sel.appendChild(opt);
    });

    const badge = $('#reviewBadge');
    const latest = batches[0];
    if(latest && latest.n_bad > 0){ badge.style.display = 'inline-block'; badge.textContent = latest.n_bad; }
    else badge.style.display = 'none';

    if(selectLatest && latest){
        sel.value = latest.id;
        await loadBatch(latest.id);
    } else if(currentBatchId){
        sel.value = currentBatchId;
    }
}

async function loadBatch(batchId){
    currentBatchId = batchId;
    try {
        const d = await api('/api/batches/' + batchId);
        scans = d.scans || [];
    } catch(e){ scans = []; }
    selectedScanId = null;
    renderScanList();
    if(currentView === 'review') renderReviewView();
}

function scanStatusKey(s){
    if(s.status === 'error') return 'error';
    if(s.bad_scan === 'bad_scan') return 'bad';
    if(s.bad_scan === 'no_bad_scan') return 'clean';
    return 'unknown';
}

function filteredScans(){
    if(scanFilter === 'all') return scans;
    if(scanFilter === 'unreviewed') return scans.filter(s => !s.review_verdict);
    return scans.filter(s => scanStatusKey(s) === 'bad' || s.review_verdict === 'false_negative');
}

function renderScanList(){
    const list = $('#fileList');
    const header = $('#fileListHeader');
    const items = filteredScans();
    header.textContent = 'Scans (' + items.length + '/' + scans.length + ')';
    list.innerHTML = '';

    if(!scans.length){
        list.innerHTML = '<div class="empty-side">No scans in this batch yet.<br>Run a pass or pick another batch.</div>';
        return;
    }
    if(!items.length){
        list.innerHTML = '<div class="empty-side">Nothing matches this filter.</div>';
        return;
    }

    items.forEach(s => {
        const key = scanStatusKey(s);
        const el = document.createElement('div');
        el.className = 'tree-file' + (selectedScanId === s.id ? ' active' : '');
        let badge = '';
        if(s.review_verdict){
            badge = '<span class="file-badge badge-reviewed">' + esc(VERDICT_LABELS[s.review_verdict] || 'reviewed') + '</span>';
        } else if(key === 'bad'){ badge = '<span class="file-badge badge-bad">bad scan</span>'; }
        else if(key === 'clean'){ badge = '<span class="file-badge badge-clean">clean</span>'; }
        else if(key === 'error'){ badge = '<span class="file-badge badge-error">error</span>'; }
        else { badge = '<span class="file-badge badge-unknown">?</span>'; }
        const inds = '<span class="stage-inds">'
            + '<span class="stage-ind' + (s.bad_scan ? ' run' : '') + '" title="Bad-scan model ' + (s.bad_scan ? 'run' : 'not run') + '">M</span>'
            + '<span class="stage-ind' + (s.page_type ? ' run' : '') + '" title="Page type ' + (s.page_type ? 'detected' : 'not detected') + '">P</span>'
            + '<span class="stage-ind' + (s.transcription ? ' run' : '') + '" title="Transcription ' + (s.transcription ? 'done' : 'not done') + '">T</span>'
            + '</span>';
        el.innerHTML = '<span class="file-dot dot-' + key + '"></span>'
            + '<span class="tree-file-name">' + esc(s.filename) + '</span>' + inds + badge;
        el.onclick = () => { selectedScanId = s.id; renderScanList(); showScan(s.id); };
        list.appendChild(el);
    });
}

/* ================= Review view ================= */
function renderReviewView(){
    topbar.style.display = 'none';
    if(!batches.length){
        content.innerHTML = '<div class="empty-state">'
            + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>'
            + '<p>No batches yet</p>'
            + '<span class="hint">Add monitored directories in Settings, then run a pass (or wait for the schedule).</span></div>';
        return;
    }
    if(selectedScanId){ showScan(selectedScanId); return; }
    const flagged = scans.filter(s => scanStatusKey(s) === 'bad');
    const unreviewedFlagged = flagged.filter(s => !s.review_verdict);
    content.innerHTML = '<div class="empty-state">'
        + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>'
        + '<p>' + flagged.length + ' flagged scan' + (flagged.length === 1 ? '' : 's') + ' in this batch</p>'
        + '<span class="hint">' + unreviewedFlagged.length + ' awaiting review — select a scan on the left.'
        + '<br>Every verdict becomes training data for the next model version.</span></div>';
}

/* ================= Bad Scans view (aggregate queue across passes) ================= */
async function renderBadScansView(){
    topbar.style.display = 'none';
    $('#fileListHeader').textContent = 'Bad Scans';
    $('#fileList').innerHTML = '<div class="empty-side">Loading…</div>';
    try { scans = (await api('/api/bad-scans')).scans || []; }
    catch(e){ scans = []; }
    selectedScanId = null;
    scanFilter = 'all';
    const badge = $('#badBadge');
    const pending = scans.filter(s => !s.review_verdict).length;
    if(pending > 0){ badge.style.display = 'inline-block'; badge.textContent = pending; }
    else badge.style.display = 'none';
    renderScanList();
    if(!scans.length){
        content.innerHTML = '<div class="empty-state">'
            + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            + '<p>No bad scans yet</p><span class="hint">Bad scans flagged by the model (or marked as missed during review) collect here across all passes — kept as training data for the bad-scan model.</span></div>';
        return;
    }
    content.innerHTML = '<div class="empty-state">'
        + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/></svg>'
        + '<p>' + scans.length + ' bad scan' + (scans.length === 1 ? '' : 's') + ' across all passes</p>'
        + '<span class="hint">Select one on the left to review. Confirming or correcting these feeds the next bad-scan model.</span></div>';
}

/* ================= Datasets view (versioned training snapshots) ================= */
let dsSelected = null;
let dsClassFilter = null;

async function renderDatasetsView(){
    topbar.style.display = 'none';
    let d;
    try { d = await api('/api/models'); } catch(e){ content.innerHTML = '<div class="panel">' + esc(e.message) + '</div>'; return; }
    const running = d.train_running;

    let html = '';
    [['badscan','Bad-scan'],['pagetype','Page-type']].forEach(([role, title]) => {
        const ds = (d.datasets || {})[role] || {counts:{}, total:0, trainable:false, min_per_class:4};
        const countsStr = Object.entries(ds.counts).map(([k,v]) => esc(k)+': <b>'+v+'</b>').join(' · ') || 'none';
        const srcStr = [ds.human?ds.human+' reviewed':'', ds.curated?ds.curated+' curated':'', ds.llm?ds.llm+' LLM':'', ds.synthetic?ds.synthetic+' synthetic':'']
            .filter(Boolean).join(', ');
        html += '<div class="panel"><h2>' + title + ' dataset'
            + '<span class="spacer"></span>'
            + (role === 'badscan'
               ? '<button class="mini-btn" onclick="genSynthetic()"' + (running ? ' disabled' : '') + '>Add synthetic bad scans</button>'
               : '')
            + '<button class="mini-btn primary" onclick="buildDataset(\'' + role + '\')"' + ((running || !ds.trainable) ? ' disabled' : '') + '>Build snapshot</button>'
            + '</h2>'
            + '<div class="panel-sub">Live labeled pool: ' + countsStr + ' (' + ds.total + ' total'
            + (srcStr ? ' — ' + srcStr : '') + (ds.trainable ? '' : '; need &ge; ' + ds.min_per_class + ' per class') + ').'
            + ' A snapshot freezes these into an immutable version you can browse, train from, and reproduce.</div>';

        const versions = (d.dataset_versions || []).filter(v => v.role === role);
        if(versions.length){
            html += '<div style="overflow-x:auto"><table class="data"><thead><tr>'
                + '<th>Version</th><th>Created</th><th>Total</th><th>Classes</th><th></th></tr></thead><tbody>';
            versions.forEach(v => {
                let counts = {}; try { counts = JSON.parse(v.counts||'{}'); } catch(e){}
                const cstr = Object.entries(counts).map(([k,x]) => esc(k)+':'+x).join(', ');
                html += '<tr><td><b>' + esc(v.version) + '</b></td>'
                    + '<td>' + fmtTime(v.created_at) + '</td><td>' + v.total + '</td>'
                    + '<td style="font-size:.72rem">' + esc(cstr) + '</td>'
                    + '<td style="white-space:nowrap">'
                    + '<button class="mini-btn" onclick="viewDataset(' + v.id + ')">Browse</button> '
                    + '<button class="mini-btn primary" onclick="trainFromDataset(\'' + role + '\',' + v.id + ')"' + (running ? ' disabled' : '') + '>Train</button> '
                    + '<button class="mini-btn danger" onclick="deleteDataset(' + v.id + ')">Delete</button>'
                    + '</td></tr>';
            });
            html += '</tbody></table></div>';
        } else {
            html += '<div class="panel-sub">No snapshots yet.</div>';
        }
        html += '</div>';
    });

    html += '<div class="panel" id="dsBrowse" style="display:none"></div>';
    content.innerHTML = html;
    if(dsSelected) viewDataset(dsSelected);
}

window.viewDataset = async function(datasetId){
    dsSelected = datasetId;
    const box = document.getElementById('dsBrowse');
    if(!box) return;
    box.style.display = 'block';
    box.innerHTML = '<div class="panel-sub">Loading dataset…</div>';
    let d;
    try { d = await api('/api/datasets/' + datasetId); } catch(e){ box.innerHTML = esc(e.message); return; }
    const items = d.items || [];
    const classes = [...new Set(items.map(i => i.label))];
    const shown = dsClassFilter ? items.filter(i => i.label === dsClassFilter) : items;
    let html = '<h2>' + esc(d.dataset.version) + ' <span class="panel-sub" style="margin:0">'
        + items.length + ' images</span></h2>';
    html += '<div class="ds-classbar">'
        + '<span class="ds-classchip' + (!dsClassFilter ? ' active' : '') + '" onclick="dsFilter(null)">all (' + items.length + ')</span>'
        + classes.map(c => '<span class="ds-classchip' + (dsClassFilter === c ? ' active' : '')
            + '" onclick="dsFilter(\'' + esc(c) + '\')">' + esc(c) + ' (' + items.filter(i=>i.label===c).length + ')</span>').join('')
        + '</div>';
    html += '<div class="ds-grid">' + shown.slice(0, 300).map(i =>
        '<div class="ds-thumb"><img loading="lazy" src="/api/dataset-items/' + i.id + '/image">'
        + '<div class="cap"><span class="lbl">' + esc(i.label) + '</span><span class="src">' + esc(i.source || '') + '/' + esc(i.split) + '</span></div></div>'
    ).join('') + '</div>';
    if(shown.length > 300) html += '<div class="panel-sub">Showing first 300 of ' + shown.length + '.</div>';
    box.innerHTML = html;
    box.scrollIntoView({behavior:'smooth', block:'nearest'});
};

window.dsFilter = function(cls){ dsClassFilter = cls; viewDataset(dsSelected); };

window.buildDataset = async function(role){
    try { await apiPost('/api/datasets/build', {role}); renderDatasetsView(); }
    catch(e){ alert(e.message); }
};

window.genSynthetic = async function(){
    if(!confirm('Generate a fresh batch of synthetic degraded bad-scan examples from your monitored images? These are kept permanently and used for training.')) return;
    try { await apiPost('/api/synthetic/badscan', {}); renderDatasetsView(); }
    catch(e){ alert(e.message); }
};

window.trainFromDataset = async function(role, datasetId){
    if(!confirm('Train a new ' + role + ' model from this dataset version?')) return;
    try { await apiPost('/api/train', {role, dataset_id: datasetId}); await refreshState(); renderDatasetsView(); }
    catch(e){ alert(e.message); }
};

window.deleteDataset = async function(datasetId){
    if(!confirm('Delete this dataset snapshot? (Training runs that used it keep their reference.)')) return;
    try { await api('/api/datasets/' + datasetId, {method:'DELETE'}); if(dsSelected===datasetId) dsSelected=null; renderDatasetsView(); }
    catch(e){ alert(e.message); }
};

function showScan(scanId){
    const s = scans.find(x => x.id === scanId);
    if(!s) return;
    renderScanDetail(s);
}

/* Re-render whichever context (Review batch list or Directories tree)
   is currently displaying this scan. */
function refreshScanDetail(scanId){
    if(currentView === 'directories'){
        if(dirvCurrentScan && dirvCurrentScan.id === scanId) renderScanDetail(dirvCurrentScan);
    } else {
        showScan(scanId);
    }
}

function renderScanDetail(s){
    const dirvMode = currentView === 'directories';
    const key = scanStatusKey(s);

    topbar.style.display = 'flex';
    topFilename.innerHTML = '<span class="topbar-dir">' + esc(s.directory) + '/</span>'
        + '<span class="topbar-name">' + esc(s.filename) + '</span>';

    const stClass = key === 'bad' ? 'status-fail' : key === 'clean' ? 'status-pass' : 'status-pending';
    topStatus.className = 'topbar-status ' + stClass;
    topStatus.textContent = s.review_verdict
        ? (VERDICT_LABELS[s.review_verdict] || s.review_verdict).toUpperCase()
        : key === 'bad' ? 'BAD SCAN' : key === 'clean' ? 'CLEAN' : key.toUpperCase();

    topActions.innerHTML = '';

    /* prev / next navigation: across the filtered batch list in Review,
       or across the processed files of the folder tree in Directories */
    let navIdx, navLen, navFn;
    if(dirvMode){
        const flat = dirvFilesFlat();
        navIdx = flat.findIndex(x => dirvSelected && x.path === dirvSelected.path);
        navLen = flat.length;
        navFn = dirvNavigate;
    } else {
        const navItems = filteredScans();
        navIdx = navItems.findIndex(x => x.id === s.id);
        navLen = navItems.length;
        navFn = navigateScan;
    }
    topNav.innerHTML = '';
    const prevBtn = document.createElement('button');
    prevBtn.className = 'btn btn-nav';
    prevBtn.innerHTML = '&#8249; Prev';
    prevBtn.title = 'Previous image (Left arrow)';
    prevBtn.disabled = navIdx <= 0;
    prevBtn.onclick = () => navFn(-1);
    const nextBtn = document.createElement('button');
    nextBtn.className = 'btn btn-nav';
    nextBtn.innerHTML = 'Next &#8250;';
    nextBtn.title = 'Next image (Right arrow)';
    nextBtn.disabled = navIdx < 0 || navIdx >= navLen - 1;
    nextBtn.onclick = () => navFn(1);
    topNav.appendChild(prevBtn);
    topNav.appendChild(nextBtn);

    if(s.status !== 'error'){
        if(key === 'bad'){
            addAction('Confirm Bad Scan', 'btn-accept', () => submitReview(s.id, 'true_positive'));
            addAction('False Positive', 'btn-ignore', () => submitReview(s.id, 'false_positive'));
        } else {
            addAction('Missed Bad Scan', 'btn-flag', () => submitReview(s.id, 'false_negative'));
            addAction('Looks Good', 'btn-good', () => submitReview(s.id, 'true_negative'));
        }
    }

    /* --- pass versions placeholder (populated async) --- */
    const versionsBox = '<div class="versions-card" id="versionsBox" style="display:none"></div>';

    /* --- check cards --- */
    const badStatus = s.bad_scan === 'bad_scan' ? 'fail' : s.bad_scan === 'no_bad_scan' ? 'pass' : 'unknown';
    const mConf = Math.round((s.bad_scan_conf || 0) * 100);
    const pConf = Math.round((s.page_type_conf || 0) * 100);
    const pageStatus = s.page_type ? 'pass' : 'unknown';
    const badReady = state && state.model_ready && state.model_ready.badscan;
    const pageReady = state && state.model_ready && state.model_ready.pagetype;

    let html = '<div class="checks-grid">';
    html += '<div class="check-card"><div class="check-header"><span class="check-dot ' + badStatus + '"></span><span class="check-name">Bad scan</span></div>'
        + '<div class="check-detail">' + (s.bad_scan ? ('Model says: <b>' + esc(s.bad_scan.replace(/_/g,' ')) + '</b> (' + mConf + '%)')
            : (badReady ? 'No prediction' : 'No trained bad-scan model active yet — verdicts you give here train it')) + '</div>'
        + '<div class="check-conf"><div class="check-conf-fill conf-' + badStatus + '" style="width:' + mConf + '%"></div></div></div>';
    let pageDetail;
    if(s.page_type){
        const src = s.page_type_source || '';
        const srcLabel = src === 'model' ? ' (' + pConf + '%, local model)'
            : src.startsWith('llm:') ? ' <span class="transcript-backend">via ' + esc(src.slice(4)) + '</span>'
            : '';
        pageDetail = '<b>' + esc(PAGE_LABELS[s.page_type] || s.page_type) + '</b>' + srcLabel;
    } else if(pageReady){
        pageDetail = 'No prediction';
    } else {
        const backendOff = !state || (state.settings || {}).transcription_backend === 'off';
        pageDetail = backendOff
            ? 'No trained page-type model yet. Enable an SGLang/Vertex backend in Settings to classify pages via LLM until one is trained.'
            : 'No trained page-type model yet — LLM classification did not return a label for this scan.';
    }
    html += '<div class="check-card"><div class="check-header"><span class="check-dot ' + pageStatus + '"></span><span class="check-name">Page type</span></div>'
        + '<div class="check-detail">' + pageDetail + '</div>'
        + '<div class="check-conf"><div class="check-conf-fill conf-' + pageStatus + '" style="width:' + pConf + '%"></div></div></div>';
    if(s.status === 'error'){
        html += '<div class="check-card"><div class="check-header"><span class="check-dot fail"></span><span class="check-name">Processing error</span></div>'
            + '<div class="check-detail">' + esc(s.error) + '</div></div>';
    }
    html += '</div>';

    /* --- per-stage re-run buttons --- */
    const rrDisabled = (state && state.batch_running) ? ' disabled' : '';
    html += '<div class="stage-rerun-row">'
        + '<button class="mini-btn" onclick="rerunScanStages(' + s.id + ', [\'badscan\'])"' + rrDisabled + '>' + (s.bad_scan ? 'Re-run' : 'Run') + ' bad-scan model</button>'
        + '<button class="mini-btn" onclick="rerunScanStages(' + s.id + ', [\'pagetype\'])"' + rrDisabled + '>' + (s.page_type ? 'Re-run' : 'Run') + ' page detection</button>'
        + '</div>';

    /* --- review panel --- */
    const chosenPage = s.review_page_type || s.page_type || '';
    let pageOpts = '<option value="">(unknown)</option>';
    (state ? state.page_types : []).forEach(pt => {
        pageOpts += '<option value="' + pt + '"' + (pt === chosenPage ? ' selected' : '') + '>' + esc(PAGE_LABELS[pt] || pt) + '</option>';
    });
    html += '<div class="review-card">'
        + '<div class="review-field"><label>Page type (confirm or correct)</label><select id="revPageType">' + pageOpts + '</select></div>'
        + '<div class="review-field review-note"><label>Notes</label><input type="text" id="revNotes" placeholder="Optional — what did you see?" value="' + esc(s.review_notes || '') + '"></div>'
        + (s.review_verdict ? '<div class="review-saved">Saved: ' + esc(VERDICT_LABELS[s.review_verdict]) + (s.reviewed_at ? ' · ' + fmtTime(s.reviewed_at) : '') + '</div>' : '')
        + '</div>';

    /* --- transcription tab content --- */
    let transHtml;
    if(s.transcription){
        transHtml = '<div class="transcript-card"><h3>Handwriting transcription '
            + '<span class="transcript-backend">' + esc(s.transcription_backend || '') + '</span></h3>'
            + '<div class="transcript-text">' + esc(s.transcription) + '</div></div>'
            + '<div class="stage-rerun-row">'
            + '<button class="mini-btn" onclick="rerunScanStages(' + s.id + ', [\'transcription\'], true)"' + rrDisabled + '>Re-run transcription</button>'
            + '</div>';
    } else {
        const backendOff = !state || (state.settings || {}).transcription_backend === 'off';
        transHtml = '<div class="transcript-card"><h3>Handwriting transcription</h3>'
            + '<div class="check-detail">'
            + (backendOff
                ? 'No transcription. Enable an SGLang or Vertex backend in Settings first.'
                : 'Not transcribed yet — pages are transcribed automatically when classified as handwriting, or run it manually below.')
            + '</div></div>'
            + '<div class="stage-rerun-row">'
            + '<button class="mini-btn" onclick="rerunScanStages(' + s.id + ', [\'transcription\'], true)"' + (backendOff ? ' disabled' : rrDisabled) + '>Transcribe this page</button>'
            + '</div>';
    }

    /* --- tab bar with run indicators --- */
    const predRan = (s.bad_scan ? 1 : 0) + (s.page_type ? 1 : 0);
    const predInd = predRan === 2 ? ' run' : predRan === 1 ? ' partial' : '';
    const tabBar = '<div class="tab-bar">'
        + '<div class="tab' + (currentTab === 'pred' ? ' active' : '') + '" id="tabPredBtn" title="Bad-scan + page-type model outputs">'
        + 'Predictions <span class="tab-ind' + predInd + '"></span></div>'
        + '<div class="tab' + (currentTab === 'trans' ? ' active' : '') + '" id="tabTransBtn" title="Handwriting transcription">'
        + 'Transcription <span class="tab-ind' + (s.transcription ? ' run' : '') + '"></span></div>'
        + '</div>';

    /* --- image viewer (right column, full page visible) --- */
    const imgHtml = '<div class="image-card">'
        + '<div class="image-card-header"><h3>Preview</h3>'
        + '<div class="viewer-toolbar"><div class="zoom-controls">'
        + '<button onclick="doZoom(-1)" title="Zoom out">&#x2212;</button>'
        + '<span class="zoom-level" id="zoomLevel">100%</span>'
        + '<button onclick="doZoom(1)" title="Zoom in">+</button>'
        + '<button onclick="zoomReset()" title="Reset zoom">Fit</button>'
        + '</div></div></div>'
        + '<div class="image-container" id="imageContainer">'
        + '<div class="image-transform" id="imageTransform">'
        + '<img class="preview-img" id="previewImg" src="/api/scans/' + s.id + '/preview">'
        + '</div></div></div>';

    content.innerHTML = '<div class="scan-layout">'
        + '<div class="scan-col-outputs">' + tabBar
        + '<div id="tabPred" style="display:' + (currentTab === 'pred' ? 'block' : 'none') + '">' + versionsBox + html + '</div>'
        + '<div id="tabTrans" style="display:' + (currentTab === 'trans' ? 'block' : 'none') + '">' + transHtml + '</div>'
        + '</div>'
        + '<div class="scan-col-image">' + imgHtml + '</div>'
        + '</div>';

    document.getElementById('tabPredBtn').onclick = () => { currentTab = 'pred'; renderScanDetail(s); };
    document.getElementById('tabTransBtn').onclick = () => { currentTab = 'trans'; renderScanDetail(s); };

    // Restore the user's preferred transcription panel height and keep
    // tracking manual resizes (drag the bottom edge of the panel).
    const transEl = document.querySelector('#tabTrans .transcript-text');
    if(transEl){
        const saved = parseInt(localStorage.getItem('qcTranscriptHeight') || '', 10);
        if(saved && saved >= 140) transEl.style.height = saved + 'px';
        let applied = transEl.offsetHeight;
        new ResizeObserver(() => {
            const h = transEl.offsetHeight;
            if(h && Math.abs(h - applied) > 4){
                applied = h;
                localStorage.setItem('qcTranscriptHeight', String(h));
            }
        }).observe(transEl);
    }
    resetZoom();
    const img = document.getElementById('previewImg');
    img.onload = setupZoom;
    loadScanVersions(s.id);
}

async function loadScanVersions(scanId){
    let versions;
    try { versions = (await api('/api/scans/' + scanId + '/versions')).versions || []; }
    catch(e){ return; }
    const box = document.getElementById('versionsBox');
    if(!box || versions.length <= 1) return;   // only show when >1 pass exists
    let html = '<h3>Pass versions (' + versions.length + ')</h3>';
    versions.forEach(v => {
        const key = v.status === 'error' ? 'unk'
            : v.bad_scan === 'bad_scan' ? 'bad' : v.bad_scan === 'no_bad_scan' ? 'clean' : 'unk';
        const label = v.status === 'error' ? 'error'
            : v.bad_scan ? v.bad_scan.replace(/_/g,' ') : 'no prediction';
        const active = v.id === scanId ? ' active' : '';
        html += '<div class="version-row' + active + '" onclick="openScanFromDir(' + v.batch_id + ',' + v.id + ')">'
            + '<span class="vpill ' + key + '">' + esc(label) + '</span>'
            + '<span class="grow"><b>' + esc(v.batch_label) + '</b> '
            + '<span class="version-meta">' + esc(v.badscan_model_version || '—')
            + ' · ' + esc(v.page_type || '—') + (v.transcription ? ' · transcribed' : '') + '</span></span>'
            + '<span class="version-meta">' + fmtTime(v.processed_at) + '</span>'
            + '</div>';
    });
    box.innerHTML = html;
    box.style.display = 'block';
}

function addAction(label, cls, fn){
    const b = document.createElement('button');
    b.className = 'btn ' + cls;
    b.textContent = label;
    b.onclick = fn;
    topActions.appendChild(b);
}

async function submitReview(scanId, verdict){
    const pageType = ($('#revPageType') || {}).value || '';
    const notes = ($('#revNotes') || {}).value || '';
    showLoading('Saving review...');
    try {
        const d = await apiPost('/api/scans/' + scanId + '/review', {verdict, page_type: pageType, notes});
        if(currentView === 'directories'){
            dirvCurrentScan = d.scan;
            renderScanDetail(d.scan);
        } else {
            const idx = scans.findIndex(x => x.id === scanId);
            if(idx >= 0) scans[idx] = d.scan;
            renderScanList();
            advanceToNext(scanId);
        }
    } catch(e){ alert(e.message); }
    finally { hideLoading(); }
}

function navigateScan(delta){
    const items = filteredScans();
    if(!items.length) return;
    let idx = items.findIndex(x => x.id === selectedScanId);
    if(idx < 0) idx = delta > 0 ? -1 : 0;
    const next = items[idx + delta];
    if(!next) return;
    selectedScanId = next.id;
    renderScanList();
    showScan(next.id);
}

function advanceToNext(currentId){
    const items = filteredScans();
    const next = items.find(s => !s.review_verdict && s.id !== currentId);
    if(next){ selectedScanId = next.id; renderScanList(); showScan(next.id); }
    else showScan(currentId);
}

/* ================= Zoom / pan (unchanged behaviour) ================= */
let zoomScale = 1, panX = 0, panY = 0, isPanning = false, panStartX = 0, panStartY = 0;
function resetZoom(){ zoomScale = 1; panX = 0; panY = 0; applyZoom(); }
function applyZoom(){
    const el = document.getElementById('imageTransform');
    if(el) el.style.transform = 'translate(' + panX + 'px,' + panY + 'px) scale(' + zoomScale + ')';
    const lbl = document.getElementById('zoomLevel');
    if(lbl) lbl.textContent = Math.round(zoomScale * 100) + '%';
}
window.doZoom = function(dir){
    const factor = dir > 0 ? 1.2 : 1/1.2;
    zoomScale = Math.max(0.25, Math.min(20, zoomScale * factor));
    applyZoom();
};
window.zoomReset = resetZoom;
function setupZoom(){
    const container = document.getElementById('imageContainer');
    if(!container) return;
    container.onwheel = function(e){
        e.preventDefault();
        const rect = container.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const oldS = zoomScale;
        const factor = e.deltaY > 0 ? 0.9 : 1.1;
        zoomScale = Math.max(0.25, Math.min(20, zoomScale * factor));
        panX = mx - (mx - panX) * (zoomScale / oldS);
        panY = my - (my - panY) * (zoomScale / oldS);
        applyZoom();
    };
    container.onmousedown = function(e){
        if(e.button !== 0) return;
        isPanning = true;
        panStartX = e.clientX - panX; panStartY = e.clientY - panY;
        container.classList.add('panning');
        e.preventDefault();
    };
    window.addEventListener('mousemove', function(e){
        if(!isPanning) return;
        panX = e.clientX - panStartX; panY = e.clientY - panStartY;
        applyZoom();
    });
    window.addEventListener('mouseup', function(){
        if(!isPanning) return;
        isPanning = false;
        const c = document.getElementById('imageContainer');
        if(c) c.classList.remove('panning');
    });
    container.ondblclick = resetZoom;
}

/* ================= Directories view (folder tree in sidebar) ================= */
let dirvExpanded = new Set();     // "<dirId>:<rel>" keys of expanded folders
let dirvSelected = null;          // {type:'folder'|'file', dirId, rel|path}
let dirvCurrentScan = null;       // full scan record shown in Directories mode
const DIRV_FOLDER_SVG = '<svg class="tree-folder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="15" height="15"><path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>';

async function renderDirectoriesView(){
    topbar.style.display = 'none';
    $('#fileListHeader').textContent = 'Directories';
    $('#fileList').innerHTML = '<div class="empty-side">Scanning directories…</div>';
    let d;
    try { d = await api('/api/directories/listing'); }
    catch(e){ $('#fileList').innerHTML = '<div class="empty-side">' + esc(e.message) + '</div>'; return; }

    window._dirvData = d.directories || [];
    // Expand monitored roots by default on first load.
    window._dirvData.forEach(dir => {
        const rootKey = dir.id + ':.';
        if(!window._dirvSeenRoots) window._dirvSeenRoots = new Set();
        if(!window._dirvSeenRoots.has(rootKey)){
            window._dirvSeenRoots.add(rootKey);
            dirvExpanded.add(rootKey);
        }
    });
    renderDirvSidebar();
    if(dirvSelected && dirvSelected.type === 'file'){
        dirvClickFile(dirvSelected.dirId, dirvSelected.path);
    } else {
        dirvShowDetail();
    }
}

function renderDirvSidebar(){
    if(currentView !== 'directories') return;
    const dirs = window._dirvData || [];
    const list = $('#fileList');
    if(!dirs.length){
        list.innerHTML = '<div class="empty-side">No monitored directories.<br>Add the folders to sweep in Settings.</div>';
        return;
    }
    let html = '';
    dirs.forEach(dir => { html += dirvSideNode(dir.id, dir.tree, 0, true); });
    list.innerHTML = html;
}

function dirvSideNode(dirId, node, depth, isRoot){
    const key = dirId + ':' + node.rel;
    const open = dirvExpanded.has(key);
    const keyEsc = esc(key).replace(/'/g, "\\'");
    const relEsc = esc(node.rel).replace(/'/g, "\\'");
    const isSel = dirvSelected && dirvSelected.type === 'folder'
        && dirvSelected.dirId === dirId && dirvSelected.rel === node.rel;

    let html = '<div class="tree-folder-header' + (isSel ? ' dirv-side-sel' : '') + '"'
        + ' style="padding-left:' + (0.6 + depth * 0.85) + 'rem"'
        + ' title="' + esc(node.rel === '.' ? '' : node.rel) + '"'
        + ' onclick="dirvClickFolder(\'' + keyEsc + '\',' + dirId + ',\'' + relEsc + '\')">'
        + '<span class="tree-chevron' + (open ? '' : ' collapsed') + '">&#9660;</span>'
        + DIRV_FOLDER_SVG
        + '<span class="tree-folder-name">' + esc(node.name) + '</span>'
        + '<span class="tree-folder-count">' + (node.new ? node.new + ' new / ' : '') + node.total + '</span>'
        + '</div>';

    if(open){
        node.dirs.forEach(child => { html += dirvSideNode(dirId, child, depth + 1, false); });
        node.files.forEach(f => {
            const dot = f.error ? 'error' : f.processed ? 'clean' : 'unknown';
            const pathEsc = esc(f.path).replace(/'/g, "\\'");
            const fSel = dirvSelected && dirvSelected.type === 'file' && dirvSelected.path === f.path;
            html += '<div class="tree-file' + (fSel ? ' active' : '') + '"'
                + ' style="padding-left:' + (1.5 + depth * 0.85) + 'rem"'
                + ' onclick="dirvClickFile(' + dirId + ',\'' + pathEsc + '\')">'
                + '<span class="file-dot dot-' + dot + '"></span>'
                + '<span class="tree-file-name">' + esc(f.name) + '</span>'
                + '<span class="stage-inds">'
                + '<span class="stage-ind' + (f.m ? ' run' : '') + '" title="Bad-scan model">M</span>'
                + '<span class="stage-ind' + (f.p ? ' run' : '') + '" title="Page type">P</span>'
                + '<span class="stage-ind' + (f.t ? ' run' : '') + '" title="Transcription">T</span>'
                + '</span></div>';
        });
    }
    return html;
}

window.dirvClickFolder = function(key, dirId, rel){
    if(dirvExpanded.has(key)) dirvExpanded.delete(key);
    else dirvExpanded.add(key);
    dirvSelected = {type: 'folder', dirId, rel};
    renderDirvSidebar();
    dirvShowDetail();
};

window.dirvClickFile = async function(dirId, path){
    dirvSelected = {type: 'file', dirId, path};
    renderDirvSidebar();
    const dir = (window._dirvData || []).find(x => x.id === dirId);
    const f = dir ? dirvFindFile(dir.tree, path) : null;
    if(f && f.processed && f.scan_id){
        // Full review experience inline: predictions, transcription,
        // verdict buttons, viewer — no need to jump to the Review tab.
        try {
            const d = await api('/api/scans/' + f.scan_id);
            dirvCurrentScan = d.scan;
            renderScanDetail(d.scan);
            return;
        } catch(e){ /* fall through to the simple panel */ }
    }
    dirvCurrentScan = null;
    dirvShowDetail();
};

/* Flattened tree-order list of ALL images for prev/next navigation —
   unprocessed files are included; they show their preview + Run. */
function dirvFilesFlat(){
    const out = [];
    function walk(dirId, node){
        node.dirs.forEach(d => walk(dirId, d));
        node.files.forEach(f => out.push({dirId, path: f.path}));
    }
    (window._dirvData || []).forEach(dir => walk(dir.id, dir.tree));
    return out;
}

function dirvNavigate(delta){
    const flat = dirvFilesFlat();
    if(!flat.length) return;
    let idx = flat.findIndex(x => dirvSelected && x.path === dirvSelected.path);
    if(idx < 0) idx = delta > 0 ? -1 : 0;
    const next = flat[idx + delta];
    if(next) dirvClickFile(next.dirId, next.path);
}

function dirvNavButtons(){
    const flat = dirvFilesFlat();
    const idx = flat.findIndex(x => dirvSelected && x.path === dirvSelected.path);
    const prevBtn = document.createElement('button');
    prevBtn.className = 'btn btn-nav';
    prevBtn.innerHTML = '&#8249; Prev';
    prevBtn.title = 'Previous image (Left arrow)';
    prevBtn.disabled = idx <= 0;
    prevBtn.onclick = () => dirvNavigate(-1);
    const nextBtn = document.createElement('button');
    nextBtn.className = 'btn btn-nav';
    nextBtn.innerHTML = 'Next &#8250;';
    nextBtn.title = 'Next image (Right arrow)';
    nextBtn.disabled = idx < 0 || idx >= flat.length - 1;
    nextBtn.onclick = () => dirvNavigate(1);
    return [prevBtn, nextBtn];
}

function dirvUpdateFileFlags(scan){
    for(const dir of (window._dirvData || [])){
        const f = dirvFindFile(dir.tree, scan.filepath);
        if(f){
            f.m = !!scan.bad_scan;
            f.p = !!scan.page_type;
            f.t = !!scan.transcription;
            f.error = scan.status === 'error';
            return;
        }
    }
}

function dirvFindFile(node, path){
    for(const f of node.files){ if(f.path === path) return f; }
    for(const d of node.dirs){
        const found = dirvFindFile(d, path);
        if(found) return found;
    }
    return null;
}

function dirvShowDetail(){
    if(currentView !== 'directories') return;
    const dirs = window._dirvData || [];
    const running = state && state.batch_running;

    if(!dirs.length){
        content.innerHTML = '<div class="empty-state">'
            + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>'
            + '<p>No monitored directories</p><span class="hint">Add the folders to sweep in Settings.</span></div>';
        return;
    }

    /* file selected (unprocessed, or fetch fallback): preview + Run,
       with the same topbar navigation as processed scans */
    if(dirvSelected && dirvSelected.type === 'file'){
        const dir = dirs.find(x => x.id === dirvSelected.dirId);
        const f = dir ? dirvFindFile(dir.tree, dirvSelected.path) : null;
        if(f){
            const escPath = esc(f.path).replace(/'/g, "\\'");
            const parts = f.path.split('/');
            topbar.style.display = 'flex';
            topFilename.innerHTML = '<span class="topbar-dir">' + esc(parts.slice(0, -1).join('/')) + '/</span>'
                + '<span class="topbar-name">' + esc(f.name) + '</span>';
            topStatus.className = 'topbar-status ' + (f.error ? 'status-fail' : 'status-pending');
            topStatus.textContent = f.error ? 'ERROR' : 'NOT PROCESSED';
            topActions.innerHTML = '';
            topNav.innerHTML = '';
            dirvNavButtons().forEach(b => topNav.appendChild(b));
            const runBtn = document.createElement('button');
            runBtn.className = 'btn btn-blue';
            runBtn.textContent = f.processed ? 'Re-run' : 'Run';
            runBtn.disabled = !!running;
            runBtn.onclick = () => runPaths([f.path]);
            topActions.appendChild(runBtn);

            content.innerHTML = '<div class="panel">'
                + '<div class="panel-sub">'
                + (f.error ? 'Processing error on last run.'
                   : f.processed ? 'Processed — stages run: '
                       + (f.m ? 'bad-scan ' : '') + (f.p ? 'page-type ' : '') + (f.t ? 'transcription' : '')
                   : 'Not processed yet — run it to get predictions. The image below is read straight from disk; only predictions and reviews are stored in the database.')
                + '</div>'
                + '<img src="' + (f.processed
                    ? '/api/scans/' + f.scan_id + '/preview'
                    : '/api/files/preview?path=' + encodeURIComponent(f.path))
                    + '" style="max-width:100%;max-height:74vh;border-radius:6px;display:block;margin:0 auto">'
                + '</div>';
            return;
        }
    }
    topbar.style.display = 'none';

    /* folder selected: summary + run buttons; default: all roots */
    let targets;
    if(dirvSelected && dirvSelected.type === 'folder'){
        const node = _dirvNode(dirvSelected.dirId, dirvSelected.rel);
        const dir = dirs.find(x => x.id === dirvSelected.dirId);
        targets = node ? [{dirId: dirvSelected.dirId, node,
                           label: dir.path + (node.rel === '.' ? '' : '/' + node.rel)}] : [];
    } else {
        targets = dirs.map(dir => ({dirId: dir.id, node: dir.tree, label: dir.path}));
    }

    let html = '';
    targets.forEach(t => {
        const relEsc = esc(t.node.rel).replace(/'/g, "\\'");
        html += '<div class="panel"><h2><div class="dirv-header" style="flex:1">'
            + '<span class="dirv-path">' + esc(t.label) + '</span>'
            + '<span class="dirv-counts"><b>' + t.node.new + '</b> new / ' + (t.node.total - t.node.new) + ' processed / ' + t.node.total + ' total</span>'
            + '</div>'
            + '<span class="spacer"></span>'
            + '<button class="mini-btn primary" onclick="runNodeNew(' + t.dirId + ', \'' + relEsc + '\')"' + ((running || !t.node.new) ? ' disabled' : '') + '>Run new (' + t.node.new + ')</button>'
            + '<button class="mini-btn" onclick="rerunNodeAll(' + t.dirId + ', \'' + relEsc + '\')"' + ((running || !t.node.total) ? ' disabled' : '') + '>Re-run all&hellip;</button>'
            + '</h2>'
            + '<div class="panel-sub">Select an image in the tree on the left to preview it, or a folder to scope a run. Runs from here appear as new versioned batches.</div>'
            + '</div>';
    });
    content.innerHTML = html || '<div class="empty-state"><p>Select a folder or image on the left</p></div>';
}

function dirvFindNode(node, rel){
    if(node.rel === rel) return node;
    for(const d of node.dirs){
        if(rel === d.rel || rel.startsWith(d.rel + '/')){
            const found = dirvFindNode(d, rel);
            if(found) return found;
        }
    }
    return null;
}

function dirvCollectFiles(node, onlyNew){
    let files = node.files.filter(f => !onlyNew || !f.processed).map(f => f.path);
    node.dirs.forEach(d => { files = files.concat(dirvCollectFiles(d, onlyNew)); });
    return files;
}

function _dirvNode(dirId, rel){
    const dir = (window._dirvData || []).find(x => x.id === dirId);
    return dir ? dirvFindNode(dir.tree, rel) : null;
}

window.runPaths = async function(paths){
    if(!paths.length) return;
    try {
        await apiPost('/api/batches/run', {paths});
        await refreshState();
        renderDirectoriesView();
    } catch(e){ alert(e.message); }
};

window.runNodeNew = function(dirId, rel){
    const node = _dirvNode(dirId, rel);
    if(node) runPaths(dirvCollectFiles(node, true));
};
window.rerunNodeAll = function(dirId, rel){
    const node = _dirvNode(dirId, rel);
    if(!node) return;
    const paths = dirvCollectFiles(node, false);
    if(!confirm('Re-process all ' + paths.length + ' file(s) under this folder as a new pass with the active models?')) return;
    runPaths(paths);
};

window.openScanFromDir = async function(batchId, scanId){
    $('#batchSelect').value = batchId;
    await loadBatch(batchId);
    selectedScanId = scanId;
    switchView('review');
    renderScanList();
    showScan(scanId);
};

/* ================= Batches view ================= */
async function renderBatchesView(){
    topbar.style.display = 'none';
    await loadBatches(false);
    let html = '<div class="panel"><h2>Processed batches'
        + '<span class="spacer"></span>'
        + '<button class="mini-btn primary" onclick="openRunModal()"' + (state && state.batch_running ? ' disabled' : '') + '>New pass&hellip;</button>'
        + '</h2>';
    html += '<div class="panel-sub">Each pass records which model versions it ran with. After retraining, re-run a pass to compare the new model against the same files.</div>';

    if(state && state.batch_running && state.batch_progress && state.batch_progress.total !== undefined){
        const p = state.batch_progress;
        html += '<div class="panel-sub"><b>Pass in progress:</b> ' + p.done + '/' + p.total
            + (p.current ? ' — ' + esc(p.current) : '') + '</div>';
    }

    if(!batches.length){
        html += '<div class="panel-sub">No batches yet.</div></div>';
        content.innerHTML = html;
        return;
    }

    html += '<div style="overflow-x:auto"><table class="data"><thead><tr>'
        + '<th>Pass</th><th>Started</th><th>Trigger</th><th>Bad-scan model</th><th>Page model</th>'
        + '<th>Total</th><th>Flagged</th><th>Clean</th><th>Unknown</th><th>Transcribed</th><th>Errors</th><th>Status</th><th></th>'
        + '</tr></thead><tbody>';
    batches.forEach(b => {
        const pill = b.status === 'done' ? 'pill-done' : b.status === 'running' ? 'pill-running'
            : b.status === 'cancelled' ? 'pill-neutral' : 'pill-error';
        const isRunning = b.status === 'running';
        let actions;
        if(isRunning){
            actions = '<button class="mini-btn danger" onclick="cancelRunningBatch()">Cancel</button>';
        } else {
            actions = '<button class="mini-btn" onclick="openBatchInReview(' + b.id + ')">Open</button> '
                + '<button class="mini-btn" onclick="rerunBatch(' + b.id + ', \'' + esc(b.label) + '\')"' + (state && state.batch_running ? ' disabled' : '') + '>Re-run&hellip;</button> '
                + '<button class="mini-btn danger" onclick="deleteBatch(' + b.id + ', \'' + esc(b.label) + '\')">Delete</button>';
        }
        html += '<tr>'
            + '<td><b>' + esc(b.label) + '</b>' + (b.rerun_of ? ' <span class="pill pill-neutral">re-pass</span>' : '') + '</td>'
            + '<td>' + fmtTime(b.started_at) + '</td>'
            + '<td>' + esc(b.trigger) + '</td>'
            + '<td>' + esc(b.badscan_model_version || '—') + '</td>'
            + '<td>' + esc(b.pagetype_model_version || '—') + '</td>'
            + '<td>' + (b.total||0) + '</td>'
            + '<td>' + (b.n_bad||0) + '</td>'
            + '<td>' + (b.n_clean||0) + '</td>'
            + '<td>' + (b.n_unknown||0) + '</td>'
            + '<td>' + (b.n_transcribed||0) + '</td>'
            + '<td>' + (b.n_errors||0) + '</td>'
            + '<td><span class="pill ' + pill + '">' + esc(b.status) + '</span></td>'
            + '<td style="white-space:nowrap">' + actions + '</td></tr>';
    });
    html += '</tbody></table></div></div>';
    content.innerHTML = html;
}

window.openBatchInReview = async function(batchId){
    $('#batchSelect').value = batchId;
    await loadBatch(batchId);
    switchView('review');
};

/* ---- Batch re-run modal: full re-pass or in-place stage update ---- */
const rerunModal = $('#rerunModal');
let rerunTargetBatch = null;
$('#rerunModalClose').onclick = () => rerunModal.classList.remove('active');
$('#rerunModalCancel').onclick = () => rerunModal.classList.remove('active');
document.querySelectorAll('input[name="rerunMode"]').forEach(r => {
    r.onchange = () => {
        const stagesMode = document.querySelector('input[name="rerunMode"]:checked').value === 'stages';
        const box = $('#rerunStagesBox');
        box.style.opacity = stagesMode ? '1' : '.45';
        box.style.pointerEvents = stagesMode ? 'auto' : 'none';
    };
});

window.rerunBatch = function(batchId, label){
    rerunTargetBatch = batchId;
    $('#rerunModalTitle').textContent = 'Re-run ' + (label || 'batch');
    rerunModal.classList.add('active');
};

$('#rerunModalGo').onclick = async function(){
    if(rerunTargetBatch == null) return;
    const mode = document.querySelector('input[name="rerunMode"]:checked').value;
    try {
        if(mode === 'full'){
            await apiPost('/api/batches/' + rerunTargetBatch + '/rerun');
        } else {
            const stages = Array.from(document.querySelectorAll('.rerunStage'))
                .filter(cb => cb.checked).map(cb => cb.value);
            if(!stages.length){ alert('Select at least one stage.'); return; }
            await apiPost('/api/batches/' + rerunTargetBatch + '/rerun-stages', {stages});
        }
        rerunModal.classList.remove('active');
        await refreshState();
        renderBatchesView();
    } catch(e){ alert(e.message); }
};

/* ---- Per-scan stage re-run ---- */
window.rerunScanStages = async function(scanId, stages, force){
    showLoading('Re-running ' + stages.join(' + ') + '…');
    try {
        const d = await apiPost('/api/scans/' + scanId + '/rerun-stages',
            {stages, force_transcribe: !!force});
        if(currentView === 'directories'){
            dirvCurrentScan = d.scan;
            dirvUpdateFileFlags(d.scan);
            renderDirvSidebar();
            renderScanDetail(d.scan);
        } else {
            const idx = scans.findIndex(x => x.id === scanId);
            if(idx >= 0) scans[idx] = d.scan;
            renderScanList();
            showScan(scanId);
        }
    } catch(e){ alert(e.message); }
    finally { hideLoading(); }
};

window.deleteBatch = async function(batchId, label){
    let msg = 'Delete ' + label + '? Its scan records will be removed and those files become eligible for future passes again.';
    try {
        const rc = await api('/api/batches/' + batchId + '/reviews-count');
        if(rc.reviews > 0){
            msg = 'Delete ' + label + '? WARNING: this also deletes ' + rc.reviews
                + ' reviewer verdict' + (rc.reviews === 1 ? '' : 's')
                + ' (training labels) recorded on its scans.';
        }
    } catch(e){}
    if(!confirm(msg)) return;
    try {
        await api('/api/batches/' + batchId, {method:'DELETE'});
        if(currentBatchId === batchId){ currentBatchId = null; scans = []; selectedScanId = null; }
        await loadBatches(currentBatchId == null);
        renderBatchesView();
    } catch(e){ alert(e.message); }
};

/* ================= Models view ================= */
async function renderModelsView(){
    topbar.style.display = 'none';
    let d;
    try { d = await api('/api/models'); } catch(e){ content.innerHTML = '<div class="panel">' + esc(e.message) + '</div>'; return; }

    const roles = [
        {role:'badscan', title:'Bad-scan model (binary)', sub:'yolo26-nano classifier: bad_scan / no_bad_scan. Trained on ALL reviewer verdicts — confirmed and missed bad scans count as "bad_scan"; false positives and confirmed-clean count as "no_bad_scan" — plus curated real scans imported from the share and any synthetic degraded examples. Retraining always runs over the full accumulated set (dataset aggregation, not per-flag).'},
        {role:'pagetype', title:'Page-type model', sub:'Classifier: blank page / handwriting / photograph / typed text, plus any custom classes from Settings. Trains on page types you confirm during review AND on unreviewed LLM labels (weak bootstrap — your confirmations always override them). Once trained, the local model handles its classes and the LLM is only used for custom classes it has not learned yet.'},
    ];

    let html = '';
    roles.forEach(r => {
        const models = (d.models || []).filter(m => m.role === r.role);
        const ds = (d.datasets || {})[r.role] || {counts:{}, trainable:false, total:0, min_per_class:4};
        const countsStr = Object.entries(ds.counts).map(([k,v]) => esc(k) + ': <b>' + v + '</b>').join(' · ');

        html += '<div class="panel"><h2>' + esc(r.title)
            + '<span class="spacer"></span>'
            + '<label style="font-size:.72rem;font-weight:500;color:var(--gray-3)">Epochs '
            + '<input type="number" id="epochs_' + r.role + '" value="' + esc((state && state.settings.retrain_epochs) || '50') + '" min="1" max="1000" style="width:70px;padding:.3rem .4rem;border:1px solid var(--gray-4);border-radius:5px;font-family:inherit"></label>'
            + '<button class="mini-btn primary" onclick="startTraining(\'' + r.role + '\')"'
            + ((d.train_running || !ds.trainable) ? ' disabled' : '') + '>Retrain from feedback</button>'
            + '</h2>';
        html += '<div class="panel-sub">' + r.sub + '</div>';
        const srcNote = (r.role === 'pagetype' && ds.total > 0)
            ? ' — ' + (ds.human || 0) + ' reviewed, ' + (ds.llm || 0) + ' LLM-labeled'
            : '';
        html += '<div class="panel-sub">Labeled feedback available: ' + (countsStr || 'none') + ' (' + ds.total + ' total' + srcNote
            + (ds.trainable ? '' : ' — need at least ' + ds.min_per_class + ' per class to train') + ')</div>';

        if(d.train_running && d.train_progress && d.train_progress.role === r.role){
            html += '<div class="panel-sub"><span class="pill pill-running">training</span> ' + esc(d.train_progress.version || '') + ' — this can take a while; keep the server running.</div>';
        }

        if(models.length){
            html += '<div style="overflow-x:auto"><table class="data"><thead><tr>'
                + '<th>Version</th><th>Created</th><th>Base</th><th>Accuracy</th><th>Notes</th><th>Status</th><th></th></tr></thead><tbody>';
            models.forEach(m => {
                let acc = '—';
                try {
                    const met = JSON.parse(m.metrics || '{}');
                    if(met.top1_accuracy) acc = (met.top1_accuracy * 100).toFixed(1) + '%';
                } catch(e){}
                html += '<tr><td><b>' + esc(m.version) + '</b></td>'
                    + '<td>' + fmtTime(m.created_at) + '</td>'
                    + '<td>' + esc(m.base_model || '—') + '</td>'
                    + '<td>' + acc + '</td>'
                    + '<td style="max-width:340px;font-size:.72rem;color:var(--gray-2)">' + esc(m.notes || '') + '</td>'
                    + '<td>' + (m.is_active ? '<span class="pill pill-active">active</span>' : '') + '</td>'
                    + '<td>' + (m.is_active ? '' : '<button class="mini-btn" onclick="activateModel(' + m.id + ')">Activate</button>') + '</td>'
                    + '</tr>';
            });
            html += '</tbody></table></div>';
        } else {
            html += '<div class="panel-sub">No versions registered yet.</div>';
        }
        html += '</div>';
    });

    /* training history */
    html += '<div class="panel"><h2>Training runs</h2>';
    const runs = d.training_runs || [];
    if(!runs.length){ html += '<div class="panel-sub">No training runs yet.</div>'; }
    else {
        html += '<div style="overflow-x:auto"><table class="data"><thead><tr>'
            + '<th>#</th><th>Role</th><th>Started</th><th>Epochs</th><th>Dataset</th><th>Result</th><th>Status</th></tr></thead><tbody>';
        runs.forEach(t => {
            let dsStr = '';
            try {
                const ds = JSON.parse(t.dataset || '{}');
                dsStr = ds.total ? (ds.total + ' images') : '';
            } catch(e){}
            const pill = t.status === 'done' ? 'pill-done' : t.status === 'running' ? 'pill-running' : 'pill-error';
            html += '<tr><td>' + t.id + '</td><td>' + esc(t.role) + '</td><td>' + fmtTime(t.started_at) + '</td>'
                + '<td>' + t.epochs + '</td><td>' + esc(dsStr) + '</td>'
                + '<td>' + esc(t.model_version || (t.error ? t.error.slice(0, 80) : '—')) + '</td>'
                + '<td><span class="pill ' + pill + '">' + esc(t.status) + '</span></td></tr>';
        });
        html += '</tbody></table></div>';
    }
    html += '</div>';
    content.innerHTML = html;
}

window.startTraining = async function(role){
    const epochsEl = document.getElementById('epochs_' + role);
    const epochs = epochsEl ? parseInt(epochsEl.value, 10) || 50 : 50;
    if(!confirm('Retrain the ' + role + ' model from accumulated feedback (' + epochs + ' epochs)?\nThe new version will be registered' + ((state && state.settings.retrain_auto_activate === '1') ? ' and activated automatically.' : '.'))) return;
    try {
        await apiPost('/api/train', {role, epochs});
        await refreshState();
        renderModelsView();
    } catch(e){ alert(e.message); }
};

window.activateModel = async function(modelId){
    try {
        await apiPost('/api/models/' + modelId + '/activate');
        await refreshState();
        renderModelsView();
    } catch(e){ alert(e.message); }
};

/* ================= Settings view ================= */
async function renderSettingsView(){
    topbar.style.display = 'none';
    await refreshState();
    const s = state.settings;
    const dirs = state.directories || [];

    let dirHtml = '';
    if(!dirs.length) dirHtml = '<div class="panel-sub">No directories yet — add the scan share(s) the pipeline should sweep.</div>';
    dirs.forEach(dd => {
        dirHtml += '<div class="dir-item' + (dd.enabled ? '' : ' disabled') + '">'
            + '<span class="grow">' + esc(dd.path) + '</span>'
            + '<button class="mini-btn" onclick="toggleDir(' + dd.id + ',' + (dd.enabled ? 'false' : 'true') + ')">' + (dd.enabled ? 'Disable' : 'Enable') + '</button>'
            + '<button class="mini-btn danger" onclick="removeDir(' + dd.id + ')">Remove</button>'
            + '</div>';
    });

    const backend = s.transcription_backend || 'off';
    const rem = state.remote || {enabled:false};
    let remHtml;
    if(!rem.enabled){
        remHtml = '<div class="panel-sub">Not configured. Set SPARK_IP + SPARK_USERNAME + SPARK_PASSWORD (and TRAIN_SERVER_TOKEN) in .env to offload training/inference to the GPU box; the app opens its own SSH tunnel.</div>';
    } else {
        const dot = rem.reachable ? 'var(--green)' : 'var(--red)';
        remHtml = '<div class="set-row" style="gap:.5rem;margin-bottom:.5rem"><span style="width:9px;height:9px;border-radius:50%;background:' + dot + '"></span>'
            + '<b>' + (rem.reachable ? 'Connected' : 'Offline') + '</b>'
            + (rem.host ? ' <span class="panel-sub" style="margin:0">' + esc(rem.host) + (rem.use_tunnel ? ' (SSH tunnel)' : '') + '</span>' : '') + '</div>'
            + (rem.reachable
                ? '<div class="panel-sub">GPU: <b>' + esc(rem.gpu || '?') + '</b> · CUDA ' + (rem.cuda ? 'yes' : 'no')
                  + ' · models loaded: ' + ((rem.loaded_models||[]).join(', ') || 'none')
                  + '<br>Batch inference: <b>' + (rem.inference ? 'ON' : 'off') + '</b> (REMOTE_INFERENCE in .env) · training offloads automatically when connected.</div>'
                : '<div class="panel-sub" style="color:var(--red)">' + esc(rem.error || 'cannot reach server') + '</div>');
    }

    content.innerHTML = ''
    + '<div class="set-grid">'

    + '<div class="panel"><h2>GPU box (remote training &amp; inference)'
    + '<span class="spacer"></span><button class="mini-btn" onclick="testRemote()">Test connection</button></h2>'
    + '<div id="remoteStatusBox">' + remHtml + '</div></div>'

    + '<div class="panel"><h2>Monitored directories</h2>'
    + '<div class="panel-sub">Each pass sweeps these folders <b>and all their subfolders</b> for image files (' + Array.from(['tif','tiff','jpg','jpeg','png']).join(', ') + ') that have not been processed before. The qc/ cache folder is excluded.</div>'
    + dirHtml
    + '<div class="set-row" style="margin-top:.75rem">'
    + '<input type="text" id="newDirInput" placeholder="/path/to/scans" style="flex:1;padding:.55rem .7rem;border:1px solid var(--gray-4);border-radius:6px;font-size:.85rem;font-family:inherit;outline:none">'
    + '<button class="mini-btn" onclick="openBrowser(\'\')">Browse</button>'
    + '<button class="mini-btn primary" onclick="addDir()">Add</button>'
    + '</div></div>'

    + '<div class="panel"><h2>Schedule</h2>'
    + '<div class="panel-sub">The pipeline runs once a day at this time over all newly scanned files. Flagged scans are then waiting in Review the next morning.</div>'
    + '<div class="set-field"><label class="set-check"><input type="checkbox" id="schedEnabled"' + (s.schedule_enabled === '1' ? ' checked' : '') + '> Enable scheduled daily pass</label></div>'
    + '<div class="set-field"><label>Run time (local)</label><input type="time" id="schedTime" value="' + esc(s.schedule_time || '02:00') + '"></div>'
    + '<div class="set-field"><label>Retraining</label>'
    + '<label class="set-check"><input type="checkbox" id="autoActivate"' + (s.retrain_auto_activate === '1' ? ' checked' : '') + '> Auto-activate newly trained model versions</label></div>'
    + '<div class="set-field"><label>Default retrain epochs</label><input type="number" id="retrainEpochs" min="1" max="1000" value="' + esc(s.retrain_epochs || '50') + '"></div>'
    + '<button class="mini-btn primary" onclick="saveSettings()">Save</button><span class="save-note" id="saveNote1" style="display:none">Saved</span>'
    + '</div>'

    + '<div class="panel"><h2>Transcription &amp; LLM backend</h2>'
    + '<div class="panel-sub">Pages classified as handwriting are transcribed during the pass and shown in Review. '
    + 'The local page-type model handles the classes it was trained on; this backend classifies anything it can&rsquo;t — '
    + 'custom classes below, or everything while no trained model is active. Confirmed labels are worked into the '
    + 'local model at the next page-type retrain.</div>'
    + '<div class="set-field"><label>Custom page classes (comma-separated)</label>'
    + '<input type="text" id="customPageTypes" value="' + esc(s.custom_page_types || '') + '" placeholder="e.g. map, music_score, envelope"></div>'
    + '<div class="set-field"><label>Backend</label><select id="trBackend">'
    + '<option value="off"' + (backend === 'off' ? ' selected' : '') + '>Off</option>'
    + '<option value="sglang"' + (backend === 'sglang' ? ' selected' : '') + '>SGLang server (adjacent machine)</option>'
    + '<option value="vertex"' + (backend === 'vertex' ? ' selected' : '') + '>Vertex AI (Gemini)</option>'
    + '</select></div>'
    + '<div id="sglangFields" style="display:' + (backend === 'sglang' ? 'block' : 'none') + '">'
    + '<div class="set-field"><label>SGLang base URL</label><input type="text" id="sglangUrl" value="' + esc(s.sglang_base_url || '') + '" placeholder="http://192.168.1.20:30000"></div>'
    + '<div class="set-field"><label>Model name</label><input type="text" id="sglangModel" value="' + esc(s.sglang_model || 'default') + '"></div>'
    + '<div class="set-field"><label>API key (optional)</label><input type="text" id="sglangKey" value="' + esc(s.sglang_api_key || '') + '"></div>'
    + '</div>'
    + '<div id="vertexFields" style="display:' + (backend === 'vertex' ? 'block' : 'none') + '">'
    + '<div class="set-field"><label>GCP project</label><input type="text" id="vertexProject" value="' + esc(s.vertex_project || 'cultural-heritage-gemini') + '"></div>'
    + '<div class="set-field"><label>Location</label><input type="text" id="vertexLocation" value="' + esc(s.vertex_location || 'global') + '"></div>'
    + '<div class="set-field"><label>Model</label><input type="text" id="vertexModel" value="' + esc(s.vertex_model || 'gemini-3.1-flash-lite') + '"></div>'
    + '<div class="panel-sub">Uses Application Default Credentials on the QC host (<code>gcloud auth application-default login</code>).</div>'
    + '</div>'
    + '<button class="mini-btn primary" onclick="saveSettings()">Save</button><span class="save-note" id="saveNote2" style="display:none">Saved</span>'
    + '</div>'

    + '</div>';

    $('#trBackend').onchange = function(){
        $('#sglangFields').style.display = this.value === 'sglang' ? 'block' : 'none';
        $('#vertexFields').style.display = this.value === 'vertex' ? 'block' : 'none';
    };
    $('#newDirInput').onkeydown = e => { if(e.key === 'Enter') addDir(); };
}

window.saveSettings = async function(){
    const payload = {
        schedule_enabled: $('#schedEnabled').checked ? '1' : '0',
        schedule_time: $('#schedTime').value || '02:00',
        retrain_auto_activate: $('#autoActivate').checked ? '1' : '0',
        retrain_epochs: $('#retrainEpochs').value || '50',
        transcription_backend: $('#trBackend').value,
        custom_page_types: $('#customPageTypes').value,
        sglang_base_url: $('#sglangUrl').value,
        sglang_model: $('#sglangModel').value,
        sglang_api_key: $('#sglangKey').value,
        vertex_project: $('#vertexProject').value,
        vertex_location: $('#vertexLocation').value,
        vertex_model: $('#vertexModel').value,
    };
    try {
        await apiPost('/api/settings', payload);
        await refreshState();
        ['saveNote1','saveNote2'].forEach(id => {
            const el = document.getElementById(id);
            if(el){ el.style.display = 'inline'; setTimeout(() => el.style.display = 'none', 2000); }
        });
    } catch(e){ alert(e.message); }
};

window.testRemote = async function(){
    const box = document.getElementById('remoteStatusBox');
    if(box) box.innerHTML = '<div class="panel-sub">Testing…</div>';
    try {
        await api('/api/remote/status');
        await refreshState();
    } catch(e){}
    renderSettingsView();
};

window.addDir = async function(){
    const path = $('#newDirInput').value.trim();
    if(!path) return;
    try {
        await apiPost('/api/directories', {path});
        await refreshState();
        renderSettingsView();
    } catch(e){ alert(e.message); }
};

window.toggleDir = async function(id, enabled){
    try {
        await apiPost('/api/directories/' + id + '/toggle', {enabled});
        await refreshState();
        renderSettingsView();
    } catch(e){ alert(e.message); }
};

window.removeDir = async function(id){
    if(!confirm('Remove this directory from monitoring? (No files are deleted.)')) return;
    try {
        await api('/api/directories/' + id, {method:'DELETE'});
        await refreshState();
        renderSettingsView();
    } catch(e){ alert(e.message); }
};

/* ================= Folder browser modal ================= */
const folderModal = $('#folderModal');
const modalBody = $('#modalBody');
const modalPath = $('#modalPath');
const modalHint = $('#modalTiffHint');
let browsePath = '';

$('#modalClose').onclick = () => folderModal.classList.remove('active');
$('#modalCancel').onclick = () => folderModal.classList.remove('active');
$('#modalSelect').onclick = async () => {
    folderModal.classList.remove('active');
    const input = $('#newDirInput');
    if(input){ input.value = browsePath; addDir(); }
};

async function openBrowser(path){
    folderModal.classList.add('active');
    let d;
    try { d = await apiPost('/api/browse', {path: path || ''}); }
    catch(e){ modalBody.innerHTML = '<div class="browse-empty">' + esc(e.message) + '</div>'; return; }
    browsePath = d.current;
    modalPath.textContent = d.current;
    modalHint.textContent = d.tiff_count ? d.tiff_count + ' image' + (d.tiff_count > 1 ? 's' : '') : '';

    let html = '';
    if(d.parent){
        const escp = d.parent.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
        html += '<div class="browse-item browse-up" onclick="openBrowser(\'' + escp + '\')">'
            + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 18l-6-6 6-6"/></svg>'
            + '<span>.. (parent directory)</span></div>';
    }
    if(d.dirs.length === 0 && !d.parent){ html += '<div class="browse-empty">No subdirectories</div>'; }
    d.dirs.forEach(entry => {
        const escp = entry.path.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
        html += '<div class="browse-item" onclick="openBrowser(\'' + escp + '\')">'
            + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>'
            + '<span>' + esc(entry.name) + '</span></div>';
    });
    modalBody.innerHTML = html;
}
window.openBrowser = openBrowser;
</script>
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
