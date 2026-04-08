"""
Flask web application for the Yale Scanning Quality Control workflow.

Directory-based QC: select a folder of TIFFs, run inference with chosen
models, and triage flagged scans as true-positive (accept) or
false-positive (ignore) to build training data.
"""

import base64
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
    QC_SUBDIR,
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
    h, w = annotated.shape[:2]
    ref = max(h, w)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, ref / 1200)
    thickness = max(2, int(ref / 600))
    box_thickness = max(3, int(ref / 400))
    pad = max(6, int(ref / 300))

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
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
            label = f"{class_name}: {confidence:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
            ly = y1 - pad if y1 > lh + pad * 2 else y1 + lh + pad
            cv2.rectangle(annotated, (x1, ly - lh - pad), (x1 + lw + pad * 2, ly + pad), color, -1)
            cv2.putText(annotated, label, (x1 + pad, ly), font, font_scale, (255, 255, 255), thickness)

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
    result = {"path": str(watch_dir), "files": files}
    result.update(_get_dir_info())
    return jsonify(result)


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


def _get_dir_info() -> dict:
    """Return current directory metadata for the tree view."""
    import os as _os

    if watch_dir is None:
        return {"dir_name": None, "parent_path": None, "subdirs": []}

    result = {
        "dir_name": watch_dir.name or str(watch_dir),
        "parent_path": str(watch_dir.parent) if watch_dir.parent != watch_dir else None,
    }

    subdirs: list[dict] = []
    try:
        for entry in sorted(_os.scandir(watch_dir), key=lambda e: e.name.lower()):
            if entry.is_dir() and not entry.name.startswith(".") and entry.name != QC_SUBDIR:
                tc = sum(
                    1 for f in _os.scandir(entry.path)
                    if f.is_file() and Path(f.name).suffix.lower() in (".tif", ".tiff")
                )
                if tc > 0:
                    subdirs.append({"name": entry.name, "path": str(watch_dir / entry.name), "tiff_count": tc})
    except PermissionError:
        pass

    result["subdirs"] = subdirs
    return result


@app.route("/api/files")
def api_files():
    files = _refresh_file_list()
    result = {"path": str(watch_dir) if watch_dir else None, "files": files}
    result.update(_get_dir_info())
    return jsonify(result)


@app.route("/api/list-tiffs", methods=["POST"])
def api_list_tiffs():
    """List TIFF filenames in a given directory (without changing watch_dir)."""
    import os as _os

    data = request.get_json(force=True) if request.is_json else {}
    dir_path = data.get("path", "").strip()
    if not dir_path:
        return jsonify({"error": "No path provided"}), 400

    p = Path(dir_path).expanduser().resolve()
    if not p.is_dir():
        return jsonify({"error": f"Not a directory: {p}"}), 400

    tiffs = sorted(
        (e.name for e in _os.scandir(p)
         if e.is_file() and Path(e.name).suffix.lower() in (".tif", ".tiff")),
        key=str.lower,
    )
    return jsonify({"path": str(p), "tiffs": tiffs})


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


def _detection_colors_hex() -> dict[str, dict[str, str]]:
    """Return DETECTION_COLORS grouped by category, converted from BGR to hex RGB."""
    colors: dict[str, dict[str, str]] = {}
    for cat_name, cat_info in QC_CATEGORIES.items():
        if cat_info.get("type") != "detection":
            continue
        for cls in cat_info.get("classes", []):
            bgr = DETECTION_COLORS.get(cls, (0, 255, 0))
            hexc = "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])
            colors.setdefault(cat_name, {})[cls] = hexc
    return colors


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global enabled_models
    if request.method == "GET":
        all_cats = list(QC_CATEGORIES.keys())
        return jsonify({
            "enabled": enabled_models,
            "available": all_cats,
            "detection_colors": _detection_colors_hex(),
        })

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
.sidebar-header { padding:1.25rem 1.25rem 1rem; border-bottom:1px solid rgba(255,255,255,.15); }
.sidebar-header h1 { font-size:1.15rem; font-weight:700; letter-spacing:.02em; }
.sidebar-header p { font-size:.75rem; opacity:.65; margin-top:.2rem; }

.dir-section { padding:1rem 1.25rem; border-bottom:1px solid rgba(255,255,255,.15); }
.dir-section label { font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; opacity:.6; display:block; margin-bottom:.4rem; }
.dir-row { display:flex; gap:.4rem; align-items:center; }
.dir-input {
    flex:1; padding:.45rem .6rem; border:1px solid rgba(255,255,255,.25); border-radius:6px;
    background:rgba(255,255,255,.08); color:var(--white); font-size:.82rem; font-family:inherit;
    outline:none; min-width:0;
}
.dir-path-display {
    display:none; margin-top:.5rem; padding:.5rem .65rem; border-radius:6px;
    background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
    font-family:'SF Mono','Fira Code',monospace; font-size:.8rem; line-height:1.7;
    color:rgba(255,255,255,.9); word-break:break-all; white-space:normal;
}
.dir-path-display.visible { display:block; }
.dir-path-seg { opacity:.65; }
.dir-path-seg:last-child { opacity:1; font-weight:600; }
.dir-input::placeholder { color:rgba(255,255,255,.35); }
.dir-input:focus { border-color:var(--yale-blue-lighter); }
.btn-sm {
    padding:.45rem .7rem; border:none; border-radius:6px; font-size:.78rem; font-weight:600;
    cursor:pointer; font-family:inherit; white-space:nowrap;
}
.btn-open { background:var(--yale-blue-lighter); color:var(--yale-blue); }
.btn-open:hover { background:#8fc4ff; }
.btn-up { background:rgba(255,255,255,.15); color:var(--white); font-size:.9rem; padding:.35rem .55rem; line-height:1; }
.btn-up:hover { background:rgba(255,255,255,.25); }
.btn-up:disabled { opacity:.3; cursor:not-allowed; }

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
.model-check {
    display:flex; align-items:center; gap:.5rem; font-size:.82rem;
    padding:.15rem .3rem; border-radius:4px; transition:background .15s; cursor:pointer;
}
.model-check:hover { background:rgba(255,255,255,.1); }
.model-check input[type="checkbox"] { accent-color:var(--yale-blue-lighter); width:15px; height:15px; }

.action-bar { padding:.75rem 1.25rem; border-bottom:1px solid rgba(255,255,255,.15); display:flex; gap:.5rem; }
.btn-action {
    flex:1; padding:.5rem; border:1px solid rgba(255,255,255,.2); border-radius:6px;
    background:rgba(255,255,255,.08); color:var(--white); font-size:.78rem; font-weight:600;
    cursor:pointer; font-family:inherit; transition:background .15s;
}
.btn-action:hover { background:rgba(255,255,255,.18); }
.btn-action:disabled { opacity:.4; cursor:not-allowed; }

/* ---- File tree ---- */
.file-list-header {
    padding:.6rem 1.25rem; font-size:.7rem; text-transform:uppercase;
    letter-spacing:.08em; opacity:.6; border-bottom:1px solid rgba(255,255,255,.1);
}
.file-list { flex:1; overflow-y:auto; }
.tree-folder-header {
    display:flex; align-items:center; gap:.4rem; padding:.45rem 1rem .45rem .8rem;
    cursor:pointer; font-size:.82rem; user-select:none; transition:background .1s;
}
.tree-folder-header:hover { background:rgba(255,255,255,.07); }
.tree-chevron { width:14px; text-align:center; font-size:.65rem; transition:transform .15s; flex-shrink:0; opacity:.7; }
.tree-chevron.collapsed { transform:rotate(-90deg); }
.tree-folder-icon { flex-shrink:0; opacity:.7; }
.tree-folder-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:500; }
.tree-folder-count { font-size:.62rem; opacity:.45; margin-left:auto; }
.tree-folder-body.collapsed { display:none; }
.tree-file {
    display:flex; align-items:center; gap:.5rem; padding:.35rem 1.25rem .35rem 2.2rem;
    cursor:pointer; font-size:.8rem; border-left:3px solid transparent; transition:background .1s;
}
.tree-file:hover { background:rgba(255,255,255,.07); }
.tree-file.active { background:rgba(255,255,255,.12); border-left-color:var(--yale-blue-lighter); }
.file-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.dot-pending { background:#8899bb; }
.dot-clean { background:#8cc63f; }
.dot-flagged { background:var(--orange); box-shadow:0 0 6px var(--orange); }
.dot-accepted { background:var(--gray-3); opacity:.5; }
.dot-ignored { background:var(--yale-blue-lighter); opacity:.6; }
.dot-analyzing { background:var(--yale-blue-lighter); animation:pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }
.tree-file-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.file-badge {
    font-size:.58rem; text-transform:uppercase; letter-spacing:.04em; padding:.12rem .35rem;
    border-radius:4px; font-weight:600; flex-shrink:0;
}
.badge-flagged { background:var(--orange); color:var(--white); }
.badge-clean { background:rgba(140,198,63,.25); color:#8cc63f; }
.badge-accepted { background:rgba(255,255,255,.1); color:var(--gray-3); }
.badge-ignored { background:rgba(99,170,255,.15); color:var(--yale-blue-lighter); }
.tree-subdir-header {
    display:flex; align-items:center; gap:.4rem; padding:.45rem 1rem .45rem .8rem;
    cursor:pointer; font-size:.82rem; user-select:none; transition:background .1s;
}
.tree-subdir-header:hover { background:rgba(255,255,255,.07); }
.tree-subdir-body.collapsed { display:none; }
.tree-subdir-file {
    display:flex; align-items:center; gap:.5rem; padding:.3rem 1.25rem .3rem 2.6rem;
    cursor:pointer; font-size:.76rem; opacity:.65; transition:all .1s;
    border-left:3px solid transparent;
}
.tree-subdir-file:hover { background:rgba(255,255,255,.07); opacity:1; }
.tree-subdir-nav {
    margin-left:auto; padding:.15rem .4rem; border:1px solid rgba(255,255,255,.2);
    border-radius:4px; font-size:.6rem; color:rgba(255,255,255,.5); cursor:pointer;
    background:none; font-family:inherit; transition:all .15s; flex-shrink:0;
}
.tree-subdir-nav:hover { background:rgba(255,255,255,.12); color:var(--white); border-color:rgba(255,255,255,.4); }

/* ---- Main ---- */
.main { flex:1; margin-left:var(--sidebar-w); display:flex; flex-direction:column; }

.topbar {
    background:var(--white); border-bottom:1px solid var(--gray-4); padding:.85rem 1.5rem;
    display:flex; align-items:flex-start; gap:.75rem; min-height:56px; flex-wrap:wrap;
}
.topbar-title {
    font-size:.88rem; color:var(--gray-1); line-height:1.5;
    word-break:break-all; white-space:normal; flex:1 1 200px; min-width:0;
}
.topbar-dir { color:var(--gray-3); font-weight:400; font-size:.78rem; }
.topbar-name { font-weight:600; }
.topbar-status { font-size:.78rem; font-weight:600; padding:.25rem .7rem; border-radius:5px; flex-shrink:0; }
.status-pass { background:var(--green-bg); color:var(--green); }
.status-fail { background:var(--orange-bg); color:var(--orange); }
.status-pending { background:#eee; color:var(--gray-3); }
.topbar-actions { margin-left:auto; display:flex; gap:.5rem; flex-shrink:0; }
.btn {
    padding:.5rem 1.1rem; border:none; border-radius:6px; font-size:.82rem;
    font-weight:600; cursor:pointer; font-family:inherit; transition:all .15s;
}
.btn-accept { background:var(--orange); color:var(--white); }
.btn-accept:hover { background:#d4621f; }
.btn-ignore { background:var(--gray-4); color:var(--gray-2); }
.btn-ignore:hover { background:#ccc; }

.content { flex:1; padding:1.5rem 2rem; overflow-y:auto; }
.empty-state {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    height:100%; min-height:400px; color:var(--gray-3); gap:.75rem;
}
.empty-state svg { width:64px; height:64px; opacity:.35; }
.empty-state p { font-size:1rem; }
.empty-state .hint { font-size:.82rem; opacity:.7; }

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

/* ---- Check cards ---- */
.checks-grid { display:flex; gap:.75rem; flex-wrap:wrap; margin-bottom:1.5rem; }
.check-card {
    background:var(--white); border:1px solid var(--gray-4); border-radius:10px;
    padding:1rem 1.15rem; min-width:180px; flex:1; cursor:default;
    transition:box-shadow .15s, border-color .15s;
}
.check-card:hover { border-color:var(--yale-blue-lighter); box-shadow:0 2px 8px rgba(0,53,107,.1); }
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

/* ---- Image viewer ---- */
.image-card {
    background:var(--white); border:1px solid var(--gray-4); border-radius:10px; padding:1.25rem;
}
.image-card-header {
    display:flex; align-items:center; justify-content:space-between; margin-bottom:.75rem;
    flex-wrap:wrap; gap:.5rem;
}
.image-card-header h3 {
    font-size:.72rem; text-transform:uppercase; letter-spacing:.08em;
    color:var(--gray-3); display:flex; align-items:center; gap:.5rem; margin:0;
}
.det-count {
    background:var(--orange); color:var(--white); font-size:.62rem; font-weight:700;
    padding:.1rem .4rem; border-radius:8px;
}
.viewer-toolbar {
    display:flex; align-items:center; gap:.75rem; font-size:.78rem; color:var(--gray-2);
}
.toolbar-toggle {
    display:flex; align-items:center; gap:.3rem; cursor:pointer; user-select:none;
    padding:.2rem .4rem; border-radius:4px; transition:background .1s;
}
.toolbar-toggle:hover { background:var(--gray-5); }
.toolbar-toggle input { accent-color:var(--yale-blue); width:14px; height:14px; }
.zoom-controls { display:flex; align-items:center; gap:.25rem; margin-left:.5rem; }
.zoom-controls button {
    width:26px; height:26px; border:1px solid var(--gray-4); border-radius:4px;
    background:var(--white); cursor:pointer; font-size:.85rem; display:flex;
    align-items:center; justify-content:center; color:var(--gray-2); transition:all .1s;
}
.zoom-controls button:hover { background:var(--gray-5); border-color:var(--gray-3); }
.zoom-level {
    min-width:38px; text-align:center; font-size:.72rem; color:var(--gray-3);
    font-variant-numeric:tabular-nums;
}
.image-container {
    overflow:hidden; border-radius:6px; background:var(--gray-5); cursor:grab; position:relative;
}
.image-container.panning { cursor:grabbing; }
.image-transform { transform-origin:0 0; }
.image-wrap { position:relative; display:inline-block; line-height:0; }
.preview-img { display:block; max-width:100%; max-height:70vh; border-radius:6px; }
.bbox-canvas { position:absolute; top:0; left:0; pointer-events:none; }

/* ---- Folder browser modal ---- */
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
        <h1>Yale Scanning QC</h1>
        <p>Quality Control</p>
    </div>
    <div class="dir-section">
        <label>Watch Directory</label>
        <div class="dir-row">
            <button class="btn-sm btn-up" id="dirUpBtn" title="Go up one level" disabled>&uarr;</button>
            <input class="dir-input" id="dirInput" placeholder="/path/to/tiffs" spellcheck="false">
            <button class="btn-sm btn-open" id="dirBrowseBtn" title="Browse for folder">Browse</button>
        </div>
        <div class="dir-path-display" id="dirPathDisplay"></div>
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

<div class="loading-overlay" id="loadingOverlay">
    <div class="spinner-box">
        <div class="spinner"></div>
        <p id="loadingText">Analyzing...</p>
    </div>
</div>

<script>
const $ = s => document.querySelector(s);
const dirInput   = $('#dirInput');
const dirUpBtn   = $('#dirUpBtn');
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
let detectionColors = {};
let lastVersion = -1;
let dirLoaded = false;
let currentDirName = '';
let parentPath = null;
let subdirs = [];
let expandedFolders = new Set();

let subdirFiles = {};

let showBboxes = true;
let showBboxLabels = true;
let highlightCat = null;
let currentFileResults = null;

let zoomScale = 1;
let panX = 0, panY = 0;
let isPanning = false;
let panStartX = 0, panStartY = 0;

/* ---- Init ---- */
(async function init(){
    const r = await fetch('/api/settings');
    const d = await r.json();
    enabledModels = d.enabled;
    availableModels = d.available;
    detectionColors = d.detection_colors || {};
    renderModelChecks();

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

/* ---- Polling ---- */
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
                updateDirInfo(fd);
                hydrateResults(files);
                renderFiles();

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

function updateDirInfo(d){
    if(d.dir_name) currentDirName = d.dir_name;
    parentPath = d.parent_path || null;
    subdirs = d.subdirs || [];
    dirUpBtn.disabled = !parentPath;
    updatePathDisplay(d.path || dirInput.value);
}

function updatePathDisplay(fullPath){
    const el = document.getElementById('dirPathDisplay');
    if(!el || !fullPath){ if(el) el.classList.remove('visible'); return; }
    const sep = fullPath.includes('\\\\') ? '\\\\' : '/';
    const parts = fullPath.split(sep).filter(Boolean);
    if(!parts.length){ el.classList.remove('visible'); return; }
    el.innerHTML = (fullPath.startsWith('/') ? '/' : '') + parts.map(p =>
        '<span class="dir-path-seg">' + p + '</span>'
    ).join(' <span style="opacity:.35">/</span> ');
    el.classList.add('visible');
}

/* ---- Settings toggle ---- */
$('#settingsToggle').onclick = () => {
    $('#settingsBody').classList.toggle('open');
    $('#settingsArrow').classList.toggle('open');
};

function renderModelChecks(){
    const container = $('#modelChecks');
    container.innerHTML = '';
    availableModels.forEach(m => {
        const checked = enabledModels.includes(m) ? 'checked' : '';
        const label = document.createElement('label');
        label.className = 'model-check';
        label.innerHTML = `<input type="checkbox" value="${m}" ${checked}><span>${m}</span>`;
        const cb = label.querySelector('input');
        cb.onchange = () => toggleModel(m, cb.checked);
        label.addEventListener('mouseenter', () => setHighlight(m));
        label.addEventListener('mouseleave', () => setHighlight(null));
        container.appendChild(label);
    });
}

async function toggleModel(model, on){
    if(on && !enabledModels.includes(model)) enabledModels.push(model);
    if(!on) enabledModels = enabledModels.filter(m=>m!==model);
    await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:enabledModels})});
    if(selectedFile) showFile(selectedFile);
}

function setHighlight(cat){
    highlightCat = cat;
    redrawBboxes();
}

/* ---- Go-up button ---- */
dirUpBtn.onclick = () => { if(parentPath) setDirectory(parentPath); };

/* ---- Folder browser modal ---- */
const folderModal = $('#folderModal');
const modalBody   = $('#modalBody');
const modalPath   = $('#modalPath');
const modalHint   = $('#modalTiffHint');
let browsePath = '';

$('#dirBrowseBtn').onclick = () => openBrowser('');
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
        subdirFiles = {};
        updateDirInfo(d);
        hydrateResults(files);
        expandedFolders = new Set([currentDirName]);
        renderFiles();
        scanBtn.disabled = false;
        analyzeBtn.disabled = false;
        dirLoaded = true;
        lastVersion = -1;
        selectedFile = null;
        topbar.style.display = 'none';
        content.innerHTML = '';
        emptyState.style.display = files.length ? 'none' : 'flex';
        if(!files.length) content.appendChild(emptyState);
    } finally { hideLoading(); }
}

/* ---- Scan / Analyze All ---- */
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

/* ---- File tree rendering ---- */
const FOLDER_SVG = '<svg class="tree-folder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>';

function renderFiles(){
    emptyState.style.display = files.length ? 'none' : 'flex';
    fileHeader.textContent = `Files (${files.length})`;
    fileList.innerHTML = '';

    if(!files.length && !subdirs.length) return;

    if(files.length > 0){
        const isExpanded = expandedFolders.has(currentDirName);
        const folderDiv = document.createElement('div');
        folderDiv.className = 'tree-folder';

        const header = document.createElement('div');
        header.className = 'tree-folder-header';
        header.innerHTML = `<span class="tree-chevron ${isExpanded?'':'collapsed'}">&#9660;</span>${FOLDER_SVG}<span class="tree-folder-name">${currentDirName || 'root'}</span><span class="tree-folder-count">${files.length}</span>`;
        header.onclick = () => {
            if(expandedFolders.has(currentDirName)) expandedFolders.delete(currentDirName);
            else expandedFolders.add(currentDirName);
            renderFiles();
        };
        folderDiv.appendChild(header);

        const body = document.createElement('div');
        body.className = 'tree-folder-body' + (isExpanded ? '' : ' collapsed');
        files.forEach(f => {
            const el = document.createElement('div');
            el.className = 'tree-file' + (selectedFile===f.name?' active':'');
            let badge = '';
            if(f.status==='flagged')  badge='<span class="file-badge badge-flagged">flagged</span>';
            if(f.status==='clean')    badge='<span class="file-badge badge-clean">pass</span>';
            if(f.status==='accepted') badge='<span class="file-badge badge-accepted">accepted</span>';
            if(f.status==='ignored')  badge='<span class="file-badge badge-ignored">ignored</span>';
            el.innerHTML = `<span class="file-dot dot-${f.status}"></span><span class="tree-file-name">${f.name}</span>${badge}`;
            el.onclick = () => selectFile(f.name);
            body.appendChild(el);
        });
        folderDiv.appendChild(body);
        fileList.appendChild(folderDiv);
    }

    subdirs.forEach(sd => {
        const key = 'sub:' + sd.name;
        const isExp = expandedFolders.has(key);

        const wrapper = document.createElement('div');
        wrapper.className = 'tree-subdir-wrapper';

        const header = document.createElement('div');
        header.className = 'tree-subdir-header';
        header.innerHTML = '<span class="tree-chevron ' + (isExp ? '' : 'collapsed') + '">&#9660;</span>'
            + FOLDER_SVG
            + '<span class="tree-folder-name">' + sd.name + '</span>'
            + '<span class="tree-folder-count">' + sd.tiff_count + '</span>'
            + '<button class="tree-subdir-nav" title="Navigate into this folder">Open</button>';

        const navBtn = header.querySelector('.tree-subdir-nav');
        navBtn.onclick = function(e){ e.stopPropagation(); setDirectory(sd.path); };

        header.onclick = function(){
            if(expandedFolders.has(key)){
                expandedFolders.delete(key);
                renderFiles();
            } else {
                expandedFolders.add(key);
                loadSubdirFiles(sd.path, sd.name);
            }
        };
        wrapper.appendChild(header);

        if(isExp){
            const body = document.createElement('div');
            body.className = 'tree-subdir-body';
            const tiffs = subdirFiles[key] || [];
            if(tiffs.length === 0){
                const empty = document.createElement('div');
                empty.className = 'tree-subdir-file';
                empty.style.opacity = '.4';
                empty.style.fontStyle = 'italic';
                empty.textContent = 'Loading...';
                body.appendChild(empty);
            } else {
                tiffs.forEach(fname => {
                    const el = document.createElement('div');
                    el.className = 'tree-subdir-file';
                    el.innerHTML = '<span class="file-dot dot-pending"></span><span class="tree-file-name">' + fname + '</span>';
                    el.onclick = function(){ setDirectory(sd.path).then(function(){ selectFile(fname); }); };
                    body.appendChild(el);
                });
            }
            wrapper.appendChild(body);
        }
        fileList.appendChild(wrapper);
    });
}

async function loadSubdirFiles(path, name){
    const key = 'sub:' + name;
    if(subdirFiles[key]){ renderFiles(); return; }
    try {
        const r = await fetch('/api/list-tiffs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:path})});
        const d = await r.json();
        subdirFiles[key] = d.tiffs || [];
    } catch(e){
        subdirFiles[key] = [];
    }
    renderFiles();
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

/* ---- Detection color helper ---- */
function getDetColor(category, className){
    const cc = detectionColors[category];
    if(cc && cc[className]) return cc[className];
    return '#00ff00';
}

/* ---- Draw bboxes on canvas ---- */
function drawBboxes(canvas, img, results, opts){
    const ctx = canvas.getContext('2d');
    const dw = img.clientWidth;
    const dh = img.clientHeight;
    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    if(!nw || !nh) return;

    canvas.width = dw;
    canvas.height = dh;
    ctx.clearRect(0, 0, dw, dh);

    const sx = dw / nw;
    const sy = dh / nh;
    const hl = opts.highlightCategory || null;
    const labels = opts.showLabels !== false;
    const enabled = opts.enabledModels || [];

    for(const [cat, res] of Object.entries(results)){
        if(cat==='overall' || cat==='annotated_image') continue;
        if(!enabled.includes(cat)) continue;
        const dets = res.detections || [];
        if(!dets.length) continue;

        const isHl = !hl || hl === cat;
        ctx.globalAlpha = isHl ? 1.0 : 0.12;

        for(const det of dets){
            const bbox = det.bbox;
            if(!bbox || bbox.length !== 4) continue;
            const [x1,y1,x2,y2] = bbox;
            const dx = x1*sx, dy = y1*sy, bw = (x2-x1)*sx, bh = (y2-y1)*sy;
            const color = getDetColor(cat, det.class);

            ctx.strokeStyle = color;
            ctx.lineWidth = isHl ? 3 : 2;
            ctx.strokeRect(dx, dy, bw, bh);

            if(labels){
                const lbl = det.class + ': ' + (det.confidence*100).toFixed(0) + '%';
                const fontSize = Math.max(13, Math.min(18, dw / 60));
                ctx.font = 'bold ' + fontSize + 'px Inter,sans-serif';
                const tm = ctx.measureText(lbl);
                const lh = fontSize + 6;
                const ly = dy > lh + 4 ? dy - 4 : dy + bh + lh;
                ctx.fillStyle = color;
                ctx.fillRect(dx, ly - lh, tm.width + 10, lh + 4);
                ctx.fillStyle = '#ffffff';
                ctx.fillText(lbl, dx + 5, ly - 4);
            }
        }
    }
    ctx.globalAlpha = 1.0;
}

function redrawBboxes(){
    const canvas = document.getElementById('bboxCanvas');
    const img = document.getElementById('previewImg');
    if(!canvas || !img || !img.naturalWidth) return;
    if(!currentFileResults || !showBboxes){
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        return;
    }
    drawBboxes(canvas, img, currentFileResults, {
        showLabels: showBboxLabels,
        highlightCategory: highlightCat,
        enabledModels: enabledModels,
    });
}

window.toggleBboxesVisible = function(on){ showBboxes = on; redrawBboxes(); };
window.toggleLabelsVisible = function(on){ showBboxLabels = on; redrawBboxes(); };

/* ---- Zoom ---- */
function resetZoom(){
    zoomScale = 1; panX = 0; panY = 0;
    applyZoom();
}

function applyZoom(){
    const el = document.getElementById('imageTransform');
    if(el) el.style.transform = 'translate('+panX+'px,'+panY+'px) scale('+zoomScale+')';
    const lbl = document.getElementById('zoomLevel');
    if(lbl) lbl.textContent = Math.round(zoomScale*100)+'%';
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
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
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
        panStartX = e.clientX - panX;
        panStartY = e.clientY - panY;
        container.classList.add('panning');
        e.preventDefault();
    };
    window.addEventListener('mousemove', function(e){
        if(!isPanning) return;
        panX = e.clientX - panStartX;
        panY = e.clientY - panStartY;
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

/* ---- Show file ---- */
function showFile(name){
    const f = files.find(x=>x.name===name);
    const results = fileResults[name];
    currentFileResults = results;

    topbar.style.display = 'flex';
    const dirPath = dirInput.value || '';
    topFilename.innerHTML = '<span class="topbar-dir">' + dirPath + '/</span><span class="topbar-name">' + name + '</span>';

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
        content.innerHTML = '<div class="empty-state"><p>Click to analyze this file</p></div>';
        return;
    }

    let checksHtml = '<div class="checks-grid" id="checksGrid">';
    let totalDet = 0;
    for(const [cat,res] of Object.entries(results)){
        if(cat==='overall'||cat==='annotated_image') continue;
        if(!enabledModels.includes(cat)) continue;
        const s = res.status||'unknown';
        const conf = ((res.confidence||0)*100).toFixed(0);
        const dets = res.detections ? res.detections.length : 0;
        totalDet += dets;
        checksHtml += '<div class="check-card" data-category="'+cat+'">'
            + '<div class="check-header"><span class="check-dot '+s+'"></span><span class="check-name">'+cat+'</span></div>'
            + '<div class="check-detail">'+(res.details||'N/A')+(dets?' ('+dets+' detections)':'')+'</div>'
            + '<div class="check-conf"><div class="check-conf-fill conf-'+s+'" style="width:'+conf+'%"></div></div>'
            + '</div>';
    }
    checksHtml += '</div>';

    const previewUrl = '/api/file/'+encodeURIComponent(name)+'/preview';
    const boxChecked = showBboxes ? 'checked' : '';
    const lblChecked = showBboxLabels ? 'checked' : '';
    const imgHtml = '<div class="image-card">'
        + '<div class="image-card-header">'
        + '<h3>Preview' + (totalDet > 0 ? ' <span class="det-count">'+totalDet+'</span>' : '') + '</h3>'
        + '<div class="viewer-toolbar">'
        + '<label class="toolbar-toggle"><input type="checkbox" '+boxChecked+' onchange="toggleBboxesVisible(this.checked)"> Boxes</label>'
        + '<label class="toolbar-toggle"><input type="checkbox" '+lblChecked+' onchange="toggleLabelsVisible(this.checked)"> Labels</label>'
        + '<div class="zoom-controls">'
        + '<button onclick="doZoom(-1)" title="Zoom out">&#x2212;</button>'
        + '<span class="zoom-level" id="zoomLevel">100%</span>'
        + '<button onclick="doZoom(1)" title="Zoom in">+</button>'
        + '<button onclick="zoomReset()" title="Reset zoom">Fit</button>'
        + '</div></div></div>'
        + '<div class="image-container" id="imageContainer">'
        + '<div class="image-transform" id="imageTransform">'
        + '<div class="image-wrap" id="imageWrap">'
        + '<img class="preview-img" id="previewImg" src="'+previewUrl+'">'
        + '<canvas class="bbox-canvas" id="bboxCanvas"></canvas>'
        + '</div></div></div></div>';

    content.innerHTML = checksHtml + imgHtml;
    resetZoom();

    const img = document.getElementById('previewImg');
    img.onload = function(){
        redrawBboxes();
        setupZoom();
    };

    document.querySelectorAll('.check-card[data-category]').forEach(card => {
        const cat = card.getAttribute('data-category');
        card.addEventListener('mouseenter', () => setHighlight(cat));
        card.addEventListener('mouseleave', () => setHighlight(null));
    });
}

window.addEventListener('resize', () => { redrawBboxes(); });

async function doAccept(name){
    showLoading('Moving file...');
    try {
        const r = await fetch('/api/file/'+encodeURIComponent(name)+'/accept',{method:'POST'});
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
        const r = await fetch('/api/file/'+encodeURIComponent(name)+'/ignore',{method:'POST'});
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
    if(next) selectFile(next.name);
    else showFile(currentName);
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
