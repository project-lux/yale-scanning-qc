"""
Yale Scanning QC — companion training + inference server.

Runs on a GPU box (e.g. the DGX Spark). SELF-CONTAINED: imports nothing
from the rest of this project, so it can be copied on its own to the box.
Only needs `flask` and `ultralytics`.

The QC app offloads two things here:

  Training  — POST /train with a zipped ultralytics classification dataset
              + base weights + params; poll GET /jobs/<id>; download the
              trained weights from GET /jobs/<id>/weights.

  Inference — deploy a model version once with POST /models/<version>,
              then classify batches of images with POST /infer.

A single GPU lock serializes training and inference so they never
contend for the device.

Run:
    QC_TRAIN_TOKEN=secret python3 train_server.py --host 0.0.0.0 --port 8100
"""

import argparse
import io
import json
import os
import shutil
import threading
import traceback
import uuid
import zipfile
from functools import wraps
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

try:
    import cv2
except Exception:  # pragma: no cover - cv2 is pulled in by ultralytics
    cv2 = None

app = Flask(__name__)

WORK_DIR = Path(os.environ.get("QC_TRAIN_WORKDIR", "./qc-trainer-work")).resolve()
JOBS_DIR = WORK_DIR / "jobs"
MODELS_DIR = WORK_DIR / "models"
TOKEN = os.environ.get("QC_TRAIN_TOKEN", "").strip()
DEVICE = os.environ.get("QC_TRAIN_DEVICE", "").strip()  # "" = auto, else "0","cpu",...

JOBS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_gpu_lock = threading.Lock()      # serialize GPU use (train vs infer)
_jobs_lock = threading.Lock()
_jobs: dict[str, dict] = {}       # job_id -> status dict

_models_lock = threading.Lock()
_models: dict[str, object] = {}   # version -> loaded YOLO


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def require_token(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if TOKEN:
            sent = (request.headers.get("Authorization", "").replace("Bearer ", "").strip()
                    or request.headers.get("X-Auth-Token", "").strip())
            if sent != TOKEN:
                return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    info = {"ok": True, "device_env": DEVICE or "auto"}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception as e:
        info["ultralytics_error"] = str(e)
    with _models_lock:
        info["loaded_models"] = sorted(_models.keys())
    return jsonify(info)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _set_job(job_id: str, **fields):
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)
        (JOBS_DIR / job_id / "status.json").write_text(json.dumps(_jobs[job_id], default=str))


def _train_worker(job_id: str, params: dict):
    job_dir = JOBS_DIR / job_id
    ds_dir = job_dir / "dataset"
    base = job_dir / "base.pt"
    epochs = int(params.get("epochs", 50))
    imgsz = int(params.get("imgsz", 640))
    try:
        _set_job(job_id, status="waiting_gpu")
        with _gpu_lock:
            _set_job(job_id, status="running", epoch=0, epochs=epochs)
            from ultralytics import YOLO

            model = YOLO(str(base))

            def _on_epoch(trainer):
                try:
                    _set_job(job_id, epoch=int(getattr(trainer, "epoch", 0)) + 1)
                except Exception:
                    pass

            model.add_callback("on_fit_epoch_end", _on_epoch)

            kwargs = dict(data=str(ds_dir), epochs=epochs, imgsz=imgsz,
                          project=str(job_dir / "out"), name="run",
                          exist_ok=True, verbose=True)
            if DEVICE:
                kwargs["device"] = DEVICE
            results = model.train(**kwargs)

        best = job_dir / "out" / "run" / "weights" / "best.pt"
        if not best.exists():
            raise RuntimeError("training produced no best.pt")
        metrics = {}
        try:
            metrics["top1_accuracy"] = float(results.results_dict.get("metrics/accuracy_top1", 0))
        except Exception:
            pass
        _set_job(job_id, status="done", metrics=metrics, weights="best.pt")
    except Exception as e:
        traceback.print_exc()
        _set_job(job_id, status="error", error=str(e))


@app.route("/train", methods=["POST"])
@require_token
def train():
    if "dataset" not in request.files or "base" not in request.files:
        return jsonify({"error": "Need 'dataset' (zip) and 'base' (weights) files"}), 400
    try:
        params = json.loads(request.form.get("params", "{}"))
    except json.JSONDecodeError:
        return jsonify({"error": "bad params JSON"}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    (job_dir / "dataset").mkdir(parents=True, exist_ok=True)

    # unpack dataset zip into job_dir/dataset
    zbytes = request.files["dataset"].read()
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        zf.extractall(job_dir / "dataset")
    # a single top-level folder in the zip is unwrapped to the split root
    entries = [p for p in (job_dir / "dataset").iterdir() if not p.name.startswith("__")]
    if len(entries) == 1 and entries[0].is_dir() and not (entries[0].name in ("train", "val")):
        inner = entries[0]
        if (inner / "train").exists():
            for c in inner.iterdir():
                shutil.move(str(c), str(job_dir / "dataset" / c.name))
            inner.rmdir()

    request.files["base"].save(str(job_dir / "base.pt"))

    _set_job(job_id, status="queued", params=params)
    threading.Thread(target=_train_worker, args=(job_id, params), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/jobs/<job_id>")
@require_token
def job_status(job_id: str):
    with _jobs_lock:
        st = _jobs.get(job_id)
    if st is None:
        sp = JOBS_DIR / job_id / "status.json"
        if sp.exists():
            st = json.loads(sp.read_text())
        else:
            return jsonify({"error": "job not found"}), 404
    return jsonify(st)


@app.route("/jobs/<job_id>/weights")
@require_token
def job_weights(job_id: str):
    best = JOBS_DIR / job_id / "out" / "run" / "weights" / "best.pt"
    if not best.exists():
        return jsonify({"error": "weights not ready"}), 404
    return send_file(str(best), as_attachment=True, download_name="best.pt")


@app.route("/jobs/<job_id>", methods=["DELETE"])
@require_token
def job_delete(job_id: str):
    with _jobs_lock:
        _jobs.pop(job_id, None)
    shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)
    return jsonify({"deleted": job_id})


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _load_model(version: str):
    with _models_lock:
        if version in _models:
            return _models[version]
        path = MODELS_DIR / f"{version}.pt"
        if not path.exists():
            return None
        from ultralytics import YOLO
        _models[version] = YOLO(str(path))
        return _models[version]


@app.route("/models/<version>", methods=["POST"])
@require_token
def deploy_model(version: str):
    if "weights" not in request.files:
        return jsonify({"error": "Need 'weights' file"}), 400
    path = MODELS_DIR / f"{version}.pt"
    request.files["weights"].save(str(path))
    with _models_lock:
        _models.pop(version, None)  # force reload
    model = _load_model(version)
    if model is None:
        return jsonify({"error": "failed to load"}), 500
    names = model.names
    classes = list(names.values()) if isinstance(names, dict) else list(names)
    return jsonify({"loaded": version, "task": model.task, "classes": classes})


@app.route("/models")
@require_token
def list_models():
    with _models_lock:
        loaded = sorted(_models.keys())
    on_disk = sorted(p.stem for p in MODELS_DIR.glob("*.pt"))
    return jsonify({"loaded": loaded, "on_disk": on_disk})


@app.route("/infer", methods=["POST"])
@require_token
def infer():
    version = request.form.get("version", "").strip()
    if not version:
        return jsonify({"error": "missing 'version'"}), 400
    model = _load_model(version)
    if model is None:
        return jsonify({"error": f"model '{version}' not deployed"}), 404

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "no images"}), 400
    if cv2 is None:
        return jsonify({"error": "cv2 unavailable on server"}), 500

    imgs = []
    for f in files:
        arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        imgs.append(img)

    out = []
    with _gpu_lock:
        results = model(imgs, verbose=False)
    for r in results:
        if r.probs is not None:
            out.append({"class": r.names[r.probs.top1], "conf": float(r.probs.top1conf)})
        else:
            out.append({"class": None, "conf": 0.0})
    return jsonify({"version": version, "predictions": out})


def main():
    parser = argparse.ArgumentParser(description="Yale QC companion train/infer server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    print(f"Yale QC trainer on {args.host}:{args.port} — workdir {WORK_DIR} "
          f"— auth {'on' if TOKEN else 'OFF (set QC_TRAIN_TOKEN)'}")
    # threaded so status polls work while a job runs
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
