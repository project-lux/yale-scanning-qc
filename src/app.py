"""
Flask web application for the Yale Scanning Quality Control workflow.

Directory-based QC: select a folder of TIFFs, run inference with chosen
models, and triage flagged scans as true-positive (accept) or
false-positive (ignore) to build training data.
"""

import base64
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file

from .config import (
    DEFAULT_ENABLED_MODELS,
    DETECTION_COLORS,
    LAST_DIR_FILE,
    QC_CATEGORIES,
)
from .qc.analyzer import QCAnalyzer
from .scanner import (
    accept_file,
    ensure_jpeg_cached,
    ensure_qc_dirs,
    get_file_status,
    ignore_file,
    jpeg_cache_path,
    load_state,
    save_state,
    scan_tiffs,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# App state (module-level singletons)
# ---------------------------------------------------------------------------
analyzer: QCAnalyzer | None = None
watch_dir: Path | None = None
enabled_models: list[str] = list(DEFAULT_ENABLED_MODELS)
# {filename: {"status": ..., "results": ..., ...}}
file_states: dict[str, dict] = {}
_lock = threading.Lock()
_dir_version = 0  # bumped every time the file list changes
_watcher_thread: threading.Thread | None = None
_watcher_stop = threading.Event()


def init_analyzer() -> QCAnalyzer:
    global analyzer
    if analyzer is None:
        print("\n" + "=" * 50)
        print("Initializing QC Analyzer...")
        print("=" * 50)
        analyzer = QCAnalyzer(preload=True)
        print("=" * 50)
        print("QC Analyzer ready!")
        print("=" * 50 + "\n")
    return analyzer


def get_analyzer() -> QCAnalyzer:
    global analyzer
    if analyzer is None:
        return init_analyzer()
    return analyzer


# ---------------------------------------------------------------------------
# Directory watcher (background polling)
# ---------------------------------------------------------------------------

def _start_watcher():
    """Start a background thread that polls the watched directory for changes."""
    global _watcher_thread
    _watcher_stop.clear()

    if _watcher_thread is not None and _watcher_thread.is_alive():
        return

    _watcher_thread = threading.Thread(target=_watcher_loop, daemon=True)
    _watcher_thread.start()


def _stop_watcher():
    _watcher_stop.set()


def _watcher_loop():
    """Poll the watched directory every 3 seconds for new/removed TIFFs."""
    global _dir_version
    prev_names: set[str] = set()

    while not _watcher_stop.is_set():
        if watch_dir is not None and watch_dir.is_dir():
            current_names = set(scan_tiffs(watch_dir))
            if current_names != prev_names:
                new_files = current_names - prev_names
                prev_names = current_names

                with _lock:
                    changed = False
                    for name in current_names:
                        if name not in file_states:
                            fs_status = get_file_status(watch_dir, name)
                            file_states[name] = {"status": fs_status, "results": None}
                            changed = True
                    for name in list(file_states.keys()):
                        if name not in current_names and file_states[name]["status"] not in ("accepted",):
                            del file_states[name]
                            changed = True
                    if changed:
                        _dir_version += 1

                for name in new_files:
                    if _watcher_stop.is_set():
                        break
                    try:
                        ensure_jpeg_cached(watch_dir, name)
                    except Exception:
                        pass
                    _auto_analyze(name)

        _watcher_stop.wait(timeout=0.1)


def _auto_analyze(filename: str):
    """Run QC analysis on a single file in the background."""
    global _dir_version
    if watch_dir is None:
        return

    tiff_path = watch_dir / filename
    if not tiff_path.exists():
        return

    try:
        jp = ensure_jpeg_cached(watch_dir, filename)
        image = cv2.imread(str(jp))
        if image is None:
            return

        qc = get_analyzer()
        results = qc.analyze(image, categories=enabled_models)

        statuses = [
            r.get("status", "unknown")
            for r in results.values()
            if isinstance(r, dict)
        ]
        if "fail" in statuses:
            results["overall"] = "fail"
        elif "unknown" in statuses:
            results["overall"] = "partial"
        else:
            results["overall"] = "pass"

        annotated = draw_detections(image, results)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        results["annotated_image"] = base64.b64encode(buf).decode("utf-8")

        status = "flagged" if results["overall"] == "fail" else "clean"

        with _lock:
            file_states[filename] = {"status": status, "results": results}
            _dir_version += 1
            _persist_state()

        print(f"  Auto-analyzed {filename}: {status}")

    except Exception as e:
        print(f"  Error auto-analyzing {filename}: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _refresh_file_list() -> list[dict]:
    """Scan the watched directory, cache JPEGs, and return file info list."""
    global file_states
    if watch_dir is None:
        return []

    ensure_qc_dirs(watch_dir)
    tiff_names = scan_tiffs(watch_dir)

    with _lock:
        for name in tiff_names:
            if name not in file_states:
                fs_status = get_file_status(watch_dir, name)
                file_states[name] = {"status": fs_status, "results": None}

        current_tiffs = set(tiff_names)
        for name in list(file_states.keys()):
            if name not in current_tiffs and file_states[name]["status"] not in ("accepted",):
                del file_states[name]

    result = []
    for name in tiff_names:
        st = file_states.get(name, {})
        entry = {
            "name": name,
            "status": st.get("status", "pending"),
        }
        if st.get("results"):
            entry["results"] = st["results"]
        result.append(entry)

    for name in list(file_states.keys()):
        if file_states[name]["status"] == "accepted" and name not in current_tiffs:
            entry = {"name": name, "status": "accepted"}
            if file_states[name].get("results"):
                entry["results"] = file_states[name]["results"]
            result.append(entry)

    return result


def draw_detections(image: np.ndarray, results: dict) -> np.ndarray:
    """Draw detection bounding boxes on the image with labels."""
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    for category in ("obstruction", "dust"):
        cat_results = results.get(category, {})
        detections = cat_results.get("detections", [])
        for det in detections:
            class_name = det.get("class", "unknown")
            confidence = det.get("confidence", 0)
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = DETECTION_COLORS.get(class_name, (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"{class_name}: {confidence:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
            ly = y1 - 5 if y1 > 30 else y1 + lh + 5
            cv2.rectangle(annotated, (x1, ly - lh - 5), (x1 + lw + 6, ly + 5), color, -1)
            cv2.putText(annotated, label, (x1 + 3, ly), font, font_scale, (255, 255, 255), thickness)

    return annotated


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/browse", methods=["POST"])
def api_browse():
    """List directories under a given path for the in-browser folder picker."""
    import os
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

    tiff_count = sum(
        1 for entry in os.scandir(p)
        if entry.is_file() and Path(entry.name).suffix.lower() in (".tif", ".tiff")
    )

    return jsonify({
        "current": str(p),
        "parent": str(p.parent) if p != p.parent else None,
        "dirs": dirs,
        "tiff_count": tiff_count,
    })


@app.route("/api/last-directory")
def api_last_directory():
    """Return the last used directory (persisted across restarts)."""
    if LAST_DIR_FILE.exists():
        saved = LAST_DIR_FILE.read_text().strip()
        if saved and Path(saved).is_dir():
            return jsonify({"path": saved})
    return jsonify({"path": None})


@app.route("/api/set-directory", methods=["POST"])
def api_set_directory():
    global watch_dir, file_states
    data = request.get_json(force=True)
    dir_path = data.get("path", "").strip()
    if not dir_path:
        return jsonify({"error": "No path provided"}), 400

    p = Path(dir_path).expanduser().resolve()
    if not p.is_dir():
        return jsonify({"error": f"Not a valid directory: {p}"}), 400

    _stop_watcher()

    with _lock:
        watch_dir = p
        file_states = load_state(watch_dir)

    ensure_qc_dirs(watch_dir)
    _start_watcher()

    try:
        LAST_DIR_FILE.write_text(str(p))
    except OSError:
        pass

    files = _refresh_file_list()
    return jsonify({"path": str(watch_dir), "files": files})


def _persist_state():
    """Write current file_states to .qc/state.json."""
    if watch_dir is not None:
        save_state(watch_dir, file_states)


def _background_cache_jpegs():
    """Convert all TIFFs to JPEG in the background."""
    if watch_dir is None:
        return
    for name in scan_tiffs(watch_dir):
        try:
            ensure_jpeg_cached(watch_dir, name)
        except Exception as e:
            print(f"Error caching {name}: {e}")


@app.route("/api/files")
def api_files():
    files = _refresh_file_list()
    return jsonify({"path": str(watch_dir) if watch_dir else None, "files": files})


@app.route("/api/poll")
def api_poll():
    """Lightweight endpoint the frontend polls to detect directory changes."""
    return jsonify({"version": _dir_version})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    if watch_dir is None:
        return jsonify({"error": "No directory selected"}), 400
    threading.Thread(target=_background_cache_jpegs, daemon=True).start()
    files = _refresh_file_list()
    return jsonify({"path": str(watch_dir), "files": files})


@app.route("/api/file/<filename>/preview")
def api_preview(filename: str):
    if watch_dir is None:
        return jsonify({"error": "No directory selected"}), 400
    try:
        jp = ensure_jpeg_cached(watch_dir, filename)
        return send_file(jp, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/file/<filename>/analyze", methods=["POST"])
def api_analyze(filename: str):
    if watch_dir is None:
        return jsonify({"error": "No directory selected"}), 400

    tiff_path = watch_dir / filename
    if not tiff_path.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        jp = ensure_jpeg_cached(watch_dir, filename)
        image = cv2.imread(str(jp))
        if image is None:
            return jsonify({"error": "Could not load image"}), 500

        qc = get_analyzer()
        results = qc.analyze(image, categories=enabled_models)

        statuses = [
            r.get("status", "unknown")
            for r in results.values()
            if isinstance(r, dict)
        ]
        if "fail" in statuses:
            results["overall"] = "fail"
        elif "unknown" in statuses:
            results["overall"] = "partial"
        else:
            results["overall"] = "pass"

        annotated = draw_detections(image, results)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        results["annotated_image"] = base64.b64encode(buf).decode("utf-8")

        status = "flagged" if results["overall"] == "fail" else "clean"

        with _lock:
            file_states[filename] = {"status": status, "results": results}
            _persist_state()

        return jsonify({"status": status, "results": results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/file/<filename>/accept", methods=["POST"])
def api_accept(filename: str):
    if watch_dir is None:
        return jsonify({"error": "No directory selected"}), 400

    with _lock:
        state = file_states.get(filename)
        if not state or not state.get("results"):
            return jsonify({"error": "File has not been analyzed yet"}), 400
        results = state["results"]

    try:
        accept_file(watch_dir, filename, results, enabled_models)
        with _lock:
            file_states[filename] = {"status": "accepted", "results": results}
            _persist_state()
        return jsonify({"status": "accepted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/file/<filename>/ignore", methods=["POST"])
def api_ignore(filename: str):
    if watch_dir is None:
        return jsonify({"error": "No directory selected"}), 400

    with _lock:
        state = file_states.get(filename)
        if not state or not state.get("results"):
            return jsonify({"error": "File has not been analyzed yet"}), 400
        results = state["results"]

    try:
        ignore_file(watch_dir, filename, results, enabled_models)
        with _lock:
            file_states[filename] = {"status": "ignored", "results": results}
            _persist_state()
        return jsonify({"status": "ignored"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global enabled_models
    if request.method == "GET":
        all_cats = list(QC_CATEGORIES.keys())
        return jsonify({"enabled": enabled_models, "available": all_cats})

    data = request.get_json(force=True)
    new_enabled = data.get("enabled")
    if isinstance(new_enabled, list):
        enabled_models = [m for m in new_enabled if m in QC_CATEGORIES]
    return jsonify({"enabled": enabled_models})


@app.route("/api/analyze-all", methods=["POST"])
def api_analyze_all():
    """Queue analysis for all pending files (runs synchronously for now)."""
    if watch_dir is None:
        return jsonify({"error": "No directory selected"}), 400

    qc = get_analyzer()
    analyzed = []
    for name in scan_tiffs(watch_dir):
        with _lock:
            st = file_states.get(name, {}).get("status", "pending")
        if st not in ("pending",):
            continue

        tiff_path = watch_dir / name
        if not tiff_path.exists():
            continue

        try:
            jp = ensure_jpeg_cached(watch_dir, name)
            image = cv2.imread(str(jp))
            if image is None:
                continue

            results = qc.analyze(image, categories=enabled_models)
            statuses = [
                r.get("status", "unknown")
                for r in results.values()
                if isinstance(r, dict)
            ]
            if "fail" in statuses:
                results["overall"] = "fail"
            elif "unknown" in statuses:
                results["overall"] = "partial"
            else:
                results["overall"] = "pass"

            annotated = draw_detections(image, results)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            results["annotated_image"] = base64.b64encode(buf).decode("utf-8")

            status = "flagged" if results["overall"] == "fail" else "clean"
            with _lock:
                file_states[name] = {"status": status, "results": results}
            analyzed.append(name)

        except Exception as e:
            print(f"Error analyzing {name}: {e}")

    with _lock:
        _persist_state()

    files = _refresh_file_list()
    return jsonify({"analyzed": analyzed, "files": files})


# ---------------------------------------------------------------------------
# HTML Template — Yale Scanning QC
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
.sidebar-header {
    padding:1.25rem 1.25rem 1rem; border-bottom:1px solid rgba(255,255,255,.15);
}
.sidebar-header h1 { font-size:1.15rem; font-weight:700; letter-spacing:.02em; }
.sidebar-header p  { font-size:.75rem; opacity:.65; margin-top:.2rem; }

/* Directory picker */
.dir-section { padding:1rem 1.25rem; border-bottom:1px solid rgba(255,255,255,.15); }
.dir-section label { font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; opacity:.6; display:block; margin-bottom:.4rem; }
.dir-row { display:flex; gap:.4rem; }
.dir-input {
    flex:1; padding:.45rem .6rem; border:1px solid rgba(255,255,255,.25); border-radius:6px;
    background:rgba(255,255,255,.08); color:var(--white); font-size:.82rem; font-family:inherit;
    outline:none;
}
.dir-input::placeholder { color:rgba(255,255,255,.35); }
.dir-input:focus { border-color:var(--yale-blue-lighter); }
.btn-sm {
    padding:.45rem .7rem; border:none; border-radius:6px; font-size:.78rem; font-weight:600;
    cursor:pointer; font-family:inherit; white-space:nowrap;
}
.btn-open { background:var(--yale-blue-lighter); color:var(--yale-blue); }
.btn-open:hover { background:#8fc4ff; }

/* Settings */
.settings-section { padding:.75rem 1.25rem; border-bottom:1px solid rgba(255,255,255,.15); }
.settings-toggle {
    font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; opacity:.6;
    cursor:pointer; display:flex; align-items:center; gap:.4rem; user-select:none;
}
.settings-toggle .arrow { transition:transform .2s; font-size:.6rem; }
.settings-toggle .arrow.open { transform:rotate(90deg); }
.settings-body { overflow:hidden; max-height:0; transition:max-height .25s ease; }
.settings-body.open { max-height:300px; }
.model-checks { padding:.6rem 0 .2rem; display:flex; flex-direction:column; gap:.35rem; }
.model-check { display:flex; align-items:center; gap:.5rem; font-size:.82rem; }
.model-check input[type="checkbox"] { accent-color:var(--yale-blue-lighter); width:15px; height:15px; }

/* Action buttons */
.action-bar { padding:.75rem 1.25rem; border-bottom:1px solid rgba(255,255,255,.15); display:flex; gap:.5rem; }
.btn-action {
    flex:1; padding:.5rem; border:1px solid rgba(255,255,255,.2); border-radius:6px;
    background:rgba(255,255,255,.08); color:var(--white); font-size:.78rem; font-weight:600;
    cursor:pointer; font-family:inherit; transition:background .15s;
}
.btn-action:hover { background:rgba(255,255,255,.18); }
.btn-action:disabled { opacity:.4; cursor:not-allowed; }

/* File list */
.file-list-header {
    padding:.6rem 1.25rem; font-size:.7rem; text-transform:uppercase;
    letter-spacing:.08em; opacity:.6; border-bottom:1px solid rgba(255,255,255,.1);
}
.file-list { flex:1; overflow-y:auto; }
.file-item {
    display:flex; align-items:center; gap:.6rem; padding:.55rem 1.25rem;
    cursor:pointer; transition:background .1s; font-size:.82rem; border-left:3px solid transparent;
}
.file-item:hover { background:rgba(255,255,255,.07); }
.file-item.active { background:rgba(255,255,255,.12); border-left-color:var(--yale-blue-lighter); }
.file-dot {
    width:10px; height:10px; border-radius:50%; flex-shrink:0;
}
.dot-pending    { background:#8899bb; }
.dot-clean      { background:#8cc63f; }
.dot-flagged    { background:var(--orange); box-shadow:0 0 6px var(--orange); }
.dot-accepted   { background:var(--gray-3); opacity:.5; }
.dot-ignored    { background:var(--yale-blue-lighter); opacity:.6; }
.dot-analyzing  { background:var(--yale-blue-lighter); animation:pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }
.file-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.file-badge {
    font-size:.6rem; text-transform:uppercase; letter-spacing:.04em; padding:.15rem .4rem;
    border-radius:4px; font-weight:600;
}
.badge-flagged  { background:var(--orange); color:var(--white); }
.badge-clean    { background:rgba(140,198,63,.25); color:#8cc63f; }
.badge-accepted { background:rgba(255,255,255,.1); color:var(--gray-3); }
.badge-ignored  { background:rgba(99,170,255,.15); color:var(--yale-blue-lighter); }

/* ---- Main ---- */
.main { flex:1; margin-left:var(--sidebar-w); display:flex; flex-direction:column; }

/* Topbar */
.topbar {
    background:var(--white); border-bottom:1px solid var(--gray-4); padding:.85rem 2rem;
    display:flex; align-items:center; gap:1rem; min-height:56px;
}
.topbar-title { font-size:.95rem; font-weight:600; color:var(--gray-1); }
.topbar-status { font-size:.78rem; font-weight:600; padding:.25rem .7rem; border-radius:5px; }
.status-pass    { background:var(--green-bg); color:var(--green); }
.status-fail    { background:var(--orange-bg); color:var(--orange); }
.status-pending { background:#eee; color:var(--gray-3); }
.topbar-actions { margin-left:auto; display:flex; gap:.5rem; }
.btn {
    padding:.5rem 1.1rem; border:none; border-radius:6px; font-size:.82rem;
    font-weight:600; cursor:pointer; font-family:inherit; transition:all .15s;
}
.btn-accept { background:var(--orange); color:var(--white); }
.btn-accept:hover { background:#d4621f; }
.btn-ignore { background:var(--gray-4); color:var(--gray-2); }
.btn-ignore:hover { background:#ccc; }

/* Content */
.content { flex:1; padding:1.5rem 2rem; overflow-y:auto; }

.empty-state {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    height:100%; min-height:400px; color:var(--gray-3); gap:.75rem;
}
.empty-state svg { width:64px; height:64px; opacity:.35; }
.empty-state p  { font-size:1rem; }
.empty-state .hint { font-size:.82rem; opacity:.7; }

/* Loading */
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

/* Results grid */
.checks-grid { display:flex; gap:.75rem; flex-wrap:wrap; margin-bottom:1.5rem; }
.check-card {
    background:var(--white); border:1px solid var(--gray-4); border-radius:10px;
    padding:1rem 1.15rem; min-width:180px; flex:1;
}
.check-header { display:flex; align-items:center; gap:.5rem; margin-bottom:.5rem; }
.check-dot { width:10px; height:10px; border-radius:50%; }
.check-dot.pass { background:var(--green); }
.check-dot.fail { background:var(--orange); }
.check-dot.unknown { background:var(--gray-3); }
.check-name { font-size:.85rem; font-weight:600; text-transform:capitalize; }
.check-detail { font-size:.75rem; color:var(--gray-2); line-height:1.4; }
.check-conf { margin-top:.5rem; height:4px; background:var(--gray-4); border-radius:2px; overflow:hidden; }
.check-conf-fill { height:100%; border-radius:2px; }
.conf-pass { background:var(--green); }
.conf-fail { background:var(--orange); }
.conf-unknown { background:var(--gray-3); }

/* Image area */
.image-grid { display:grid; grid-template-columns:1fr 1fr; gap:1.25rem; }
.image-grid.single { grid-template-columns:1fr; max-width:800px; }
.image-card {
    background:var(--white); border:1px solid var(--gray-4); border-radius:10px; padding:1.25rem;
}
.image-card h3 {
    font-size:.72rem; text-transform:uppercase; letter-spacing:.08em;
    color:var(--gray-3); margin-bottom:.75rem; display:flex; align-items:center; gap:.5rem;
}
.det-count {
    background:var(--orange); color:var(--white); font-size:.62rem; font-weight:700;
    padding:.1rem .4rem; border-radius:8px;
}
.preview-img {
    width:100%; border-radius:6px; max-height:65vh; object-fit:contain; background:var(--gray-5);
}

/* Folder browser modal */
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
    display:flex; align-items:center; gap:.5rem;
}
.modal-path .tiff-hint {
    margin-left:auto; font-family:'Inter',sans-serif; font-size:.7rem;
    color:var(--yale-blue-light); font-weight:600;
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
    border:none; border-radius:6px; font-size:.82rem; font-weight:600;
    cursor:pointer; font-family:inherit;
}
.modal-footer .btn-select:hover { background:var(--yale-blue-light); }
.modal-footer .btn-cancel {
    padding:.5rem 1.2rem; background:var(--gray-4); color:var(--gray-2);
    border:none; border-radius:6px; font-size:.82rem; font-weight:600;
    cursor:pointer; font-family:inherit;
}
.modal-footer .btn-cancel:hover { background:#ccc; }

/* Responsive */
@media(max-width:1100px){ .image-grid{grid-template-columns:1fr;} }
</style>
</head>
<body>
<div class="app">

<!-- Sidebar -->
<aside class="sidebar">
    <div class="sidebar-header">
        <h1>Yale Scanning QC</h1>
        <p>Quality Control</p>
    </div>

    <div class="dir-section">
        <label>Watch Directory</label>
        <div class="dir-row">
            <input class="dir-input" id="dirInput" placeholder="/path/to/tiffs" spellcheck="false">
            <button class="btn-sm btn-open" id="dirBrowseBtn" title="Browse for folder">Browse</button>
        </div>
    </div>

    <div class="settings-section">
        <div class="settings-toggle" id="settingsToggle">
            <span class="arrow" id="settingsArrow">&#9654;</span> Models
        </div>
        <div class="settings-body" id="settingsBody">
            <div class="model-checks" id="modelChecks"></div>
        </div>
    </div>

    <div class="action-bar">
        <button class="btn-action" id="scanBtn" disabled>Scan</button>
        <button class="btn-action" id="analyzeAllBtn" disabled>Analyze All</button>
    </div>

    <div class="file-list-header" id="fileListHeader">Files</div>
    <div class="file-list" id="fileList"></div>
</aside>

<!-- Main -->
<div class="main">
    <div class="topbar" id="topbar" style="display:none">
        <div class="topbar-title" id="topFilename"></div>
        <div class="topbar-status" id="topStatus"></div>
        <div class="topbar-actions" id="topActions"></div>
    </div>

    <div class="content" id="content">
        <div class="empty-state" id="emptyState">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/>
            </svg>
            <p>Select a directory to begin</p>
            <span class="hint">Enter the path to a folder of TIFF scans</span>
        </div>
    </div>
</div>

</div>

<!-- Folder browser modal -->
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
            <button class="btn-select" id="modalSelect">Select This Folder</button>
        </div>
    </div>
</div>

<!-- Loading overlay -->
<div class="loading-overlay" id="loadingOverlay">
    <div class="spinner-box">
        <div class="spinner"></div>
        <p id="loadingText">Analyzing...</p>
    </div>
</div>

<script>
const $ = s => document.querySelector(s);
const dirInput   = $('#dirInput');
const dirBrowseBtn = $('#dirBrowseBtn');
const scanBtn    = $('#scanBtn');
const analyzeBtn = $('#analyzeAllBtn');
const fileList   = $('#fileList');
const fileHeader = $('#fileListHeader');
const content    = $('#content');
const emptyState = $('#emptyState');
const topbar     = $('#topbar');
const topFilename= $('#topFilename');
const topStatus  = $('#topStatus');
const topActions = $('#topActions');
const loadingOv  = $('#loadingOverlay');
const loadingTxt = $('#loadingText');

let files = [];
let selectedFile = null;
let fileResults = {};
let enabledModels = [];
let availableModels = [];
let lastVersion = -1;
let dirLoaded = false;

// -- Init --
(async function init(){
    const r = await fetch('/api/settings');
    const d = await r.json();
    enabledModels = d.enabled;
    availableModels = d.available;
    renderModelChecks();

    // Auto-load last used directory
    try {
        const lr = await fetch('/api/last-directory');
        const ld = await lr.json();
        if(ld.path){
            dirInput.value = ld.path;
            await setDirectory(ld.path);
        }
    } catch(e){}

    startPolling();
})();

// -- Directory change polling (every 3s) --
function startPolling(){
    setInterval(async ()=>{
        if(!dirLoaded) return;
        try {
            const r = await fetch('/api/poll');
            const d = await r.json();
            if(d.version !== lastVersion){
                lastVersion = d.version;
                const prevNames = new Set(files.map(f=>f.name));
                const fr = await fetch('/api/files');
                const fd = await fr.json();
                files = fd.files;
                hydrateResults(files);
                renderFiles();

                // Auto-select the newest file that was just analyzed
                const newest = files.find(f =>
                    !prevNames.has(f.name) || (f.results && !fileResults[f.name])
                );
                if(newest && newest.results){
                    fileResults[newest.name] = newest.results;
                    selectedFile = newest.name;
                    renderFiles();
                    showFile(newest.name);
                }
            }
        } catch(e){}
    }, 500);
}

// -- Settings toggle --
$('#settingsToggle').onclick = () => {
    $('#settingsBody').classList.toggle('open');
    $('#settingsArrow').classList.toggle('open');
};

function renderModelChecks(){
    const container = $('#modelChecks');
    container.innerHTML = '';
    availableModels.forEach(m => {
        const checked = enabledModels.includes(m) ? 'checked' : '';
        container.innerHTML += `<label class="model-check"><input type="checkbox" value="${m}" ${checked} onchange="toggleModel('${m}', this.checked)"><span>${m}</span></label>`;
    });
}

window.toggleModel = async function(model, on){
    if(on && !enabledModels.includes(model)) enabledModels.push(model);
    if(!on) enabledModels = enabledModels.filter(m=>m!==model);
    await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:enabledModels})});
};

// -- Folder browser modal --
const folderModal = $('#folderModal');
const modalBody   = $('#modalBody');
const modalPath   = $('#modalPath');
const modalHint   = $('#modalTiffHint');
let browsePath = '';

dirBrowseBtn.onclick = () => openBrowser('');
$('#modalClose').onclick = () => folderModal.classList.remove('active');
$('#modalCancel').onclick = () => folderModal.classList.remove('active');
$('#modalSelect').onclick = async () => {
    folderModal.classList.remove('active');
    dirInput.value = browsePath;
    await setDirectory(browsePath);
};

async function openBrowser(path){
    folderModal.classList.add('active');
    const r = await fetch('/api/browse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:path||''})});
    const d = await r.json();
    if(d.error){ modalBody.innerHTML=`<div class="browse-empty">${d.error}</div>`; return; }
    browsePath = d.current;
    modalPath.textContent = d.current;
    modalHint.textContent = d.tiff_count ? d.tiff_count+' TIFF'+(d.tiff_count>1?'s':'') : '';

    let html = '';
    if(d.parent){
        const esc = d.parent.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
        html += `<div class="browse-item browse-up" onclick="openBrowser('${esc}')">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 18l-6-6 6-6"/></svg>
            <span>.. (parent directory)</span>
        </div>`;
    }
    if(d.dirs.length===0 && !d.parent){
        html += '<div class="browse-empty">No subdirectories</div>';
    }
    d.dirs.forEach(entry => {
        const esc = entry.path.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
        html += `<div class="browse-item" onclick="openBrowser('${esc}')">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>
            <span>${entry.name}</span>
        </div>`;
    });
    modalBody.innerHTML = html;
}
window.openBrowser = openBrowser;

dirInput.onkeydown = e => { if(e.key==='Enter') setDirectory(dirInput.value.trim()); };

async function setDirectory(p){
    if(!p) return;
    showLoading('Opening directory...');
    try {
        const r = await fetch('/api/set-directory',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p})});
        const d = await r.json();
        if(d.error){ alert(d.error); return; }
        dirInput.value = d.path;
        files = d.files;
        fileResults = {};
        hydrateResults(files);
        renderFiles();
        scanBtn.disabled = false;
        analyzeBtn.disabled = false;
        dirLoaded = true;
        lastVersion = -1;
    } finally { hideLoading(); }
}

// -- Scan / Analyze All --
scanBtn.onclick = async () => {
    showLoading('Scanning directory...');
    try {
        const r = await fetch('/api/scan',{method:'POST'});
        const d = await r.json();
        files = d.files;
        hydrateResults(files);
        renderFiles();
    } finally { hideLoading(); }
};

analyzeBtn.onclick = async () => {
    showLoading('Analyzing all files...');
    try {
        const r = await fetch('/api/analyze-all',{method:'POST'});
        const d = await r.json();
        files = d.files;
        hydrateResults(files);
        renderFiles();
        if(selectedFile) showFile(selectedFile);
    } finally { hideLoading(); }
};

function hydrateResults(fileList){
    fileList.forEach(f => {
        if(f.results) fileResults[f.name] = f.results;
    });
}

// -- File list --
function renderFiles(){
    emptyState.style.display = files.length ? 'none' : 'flex';
    fileHeader.textContent = `Files (${files.length})`;
    fileList.innerHTML = '';
    files.forEach(f => {
        const el = document.createElement('div');
        el.className = 'file-item' + (selectedFile===f.name?' active':'');
        let badge = '';
        if(f.status==='flagged')  badge='<span class="file-badge badge-flagged">flagged</span>';
        if(f.status==='clean')    badge='<span class="file-badge badge-clean">pass</span>';
        if(f.status==='accepted') badge='<span class="file-badge badge-accepted">accepted</span>';
        if(f.status==='ignored')  badge='<span class="file-badge badge-ignored">ignored</span>';
        el.innerHTML = `<span class="file-dot dot-${f.status}"></span><span class="file-name">${f.name}</span>${badge}`;
        el.onclick = () => selectFile(f.name);
        fileList.appendChild(el);
    });
}

async function selectFile(name){
    selectedFile = name;
    renderFiles();
    const st = files.find(f=>f.name===name);
    if(!st || st.status==='pending'){
        await analyzeOne(name);
    } else {
        showFile(name);
    }
}

async function analyzeOne(name){
    showLoading('Analyzing ' + name + '...');
    try {
        const r = await fetch(`/api/file/${encodeURIComponent(name)}/analyze`,{method:'POST'});
        const d = await r.json();
        if(d.error){ alert(d.error); hideLoading(); return; }
        fileResults[name] = d.results;
        const idx = files.findIndex(f=>f.name===name);
        if(idx>=0) files[idx].status = d.status;
        renderFiles();
        showFile(name);
    } finally { hideLoading(); }
}

function showFile(name){
    const f = files.find(x=>x.name===name);
    const results = fileResults[name];

    topbar.style.display = 'flex';
    topFilename.textContent = name;

    const st = f ? f.status : 'pending';
    topStatus.className = 'topbar-status ' + (st==='clean'||st==='ignored'?'status-pass': st==='flagged'?'status-fail':'status-pending');
    topStatus.textContent = st.toUpperCase();

    topActions.innerHTML = '';
    if(st === 'flagged'){
        const accBtn = document.createElement('button');
        accBtn.className = 'btn btn-accept';
        accBtn.textContent = 'Accept (Rescan)';
        accBtn.onclick = () => doAccept(name);

        const ignBtn = document.createElement('button');
        ignBtn.className = 'btn btn-ignore';
        ignBtn.textContent = 'Ignore';
        ignBtn.onclick = () => doIgnore(name);

        topActions.appendChild(accBtn);
        topActions.appendChild(ignBtn);
    }

    if(!results){
        content.innerHTML = `<div class="empty-state"><p>Click to analyze this file</p></div>`;
        return;
    }

    let checksHtml = '<div class="checks-grid">';
    let totalDet = 0;
    for(const [cat,res] of Object.entries(results)){
        if(cat==='overall'||cat==='annotated_image') continue;
        const s = res.status||'unknown';
        const conf = ((res.confidence||0)*100).toFixed(0);
        const dets = res.detections ? res.detections.length : 0;
        totalDet += dets;
        checksHtml += `<div class="check-card">
            <div class="check-header"><span class="check-dot ${s}"></span><span class="check-name">${cat}</span></div>
            <div class="check-detail">${res.details||'N/A'}${dets?' ('+dets+' detections)':''}</div>
            <div class="check-conf"><div class="check-conf-fill conf-${s}" style="width:${conf}%"></div></div>
        </div>`;
    }
    checksHtml += '</div>';

    const previewUrl = `/api/file/${encodeURIComponent(name)}/preview`;
    let imgHtml = '';
    if(totalDet > 0 && results.annotated_image){
        imgHtml = `<div class="image-grid">
            <div class="image-card"><h3>Original</h3><img class="preview-img" src="${previewUrl}"></div>
            <div class="image-card"><h3>Detected Issues <span class="det-count">${totalDet}</span></h3><img class="preview-img" src="data:image/jpeg;base64,${results.annotated_image}"></div>
        </div>`;
    } else {
        imgHtml = `<div class="image-grid single"><div class="image-card"><h3>Preview</h3><img class="preview-img" src="${previewUrl}"></div></div>`;
    }

    content.innerHTML = checksHtml + imgHtml;
}

async function doAccept(name){
    showLoading('Moving file...');
    try {
        const r = await fetch(`/api/file/${encodeURIComponent(name)}/accept`,{method:'POST'});
        const d = await r.json();
        if(d.error){ alert(d.error); return; }
        const idx = files.findIndex(f=>f.name===name);
        if(idx>=0) files[idx].status = 'accepted';
        renderFiles();
        advanceToNextFlagged(name);
    } finally { hideLoading(); }
}

async function doIgnore(name){
    showLoading('Saving...');
    try {
        const r = await fetch(`/api/file/${encodeURIComponent(name)}/ignore`,{method:'POST'});
        const d = await r.json();
        if(d.error){ alert(d.error); return; }
        const idx = files.findIndex(f=>f.name===name);
        if(idx>=0) files[idx].status = 'ignored';
        renderFiles();
        advanceToNextFlagged(name);
    } finally { hideLoading(); }
}

function advanceToNextFlagged(currentName){
    const next = files.find(f=>f.status==='flagged' && f.name!==currentName);
    if(next){
        selectFile(next.name);
    } else {
        showFile(currentName);
    }
}

function showLoading(msg){ loadingTxt.textContent=msg||'Loading...'; loadingOv.classList.add('active'); }
function hideLoading(){ loadingOv.classList.remove('active'); }
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

    init_analyzer()

    print(f"\nStarting Yale Scanning QC at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
