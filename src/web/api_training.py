"""
Training API: start a run, stream its progress, list history.

Progress is streamed with Server-Sent Events. Training only ever reports
*outward* — epoch, loss, accuracy — so a one-way stream is the right shape,
and it needs no extra dependency or async worker.
"""

import json
import time

from flask import Blueprint, Response, jsonify, request

from .. import db, retrain
from ..config import MODEL_ROLES

bp = Blueprint("api_training", __name__, url_prefix="/api/training")

# How often the stream re-reads progress, and when to give up on an idle
# stream so a forgotten browser tab does not hold a worker forever.
POLL_SECONDS = 1.0
IDLE_TIMEOUT = 15 * 60


@bp.get("/state")
def state():
    """One-shot snapshot: is a run active, and what does the store hold."""
    from .. import remote

    return jsonify({
        "running": retrain.is_running(),
        "progress": retrain.progress,
        "datasets": {r: retrain.dataset_summary(r) for r in MODEL_ROLES},
        "active_models": {r: db.get_active_model(r) for r in MODEL_ROLES},
        "remote": remote.status(),
    })


@bp.get("/runs")
def runs():
    """Recent training runs, newest first."""
    limit = min(200, max(1, int(request.args.get("limit") or 25)))
    return jsonify({"runs": db.list_training_runs(limit=limit)})


@bp.get("/models")
def models():
    role = request.args.get("role")
    if role and role not in MODEL_ROLES:
        return jsonify({"error": f"Unknown role '{role}'"}), 400
    return jsonify({"models": db.list_models(role)})


@bp.post("/models/<int:model_id>/activate")
def activate(model_id: int):
    from .. import inference

    rec = db.activate_model(model_id)
    if rec is None:
        return jsonify({"error": "Model not found"}), 404
    inference.invalidate_cache()
    return jsonify({"model": rec})


@bp.post("/preview")
def preview_selection():
    """
    Show what a dataset build would choose, without building it — so the
    composition policy is inspectable before committing a version.
    """
    role = (request.get_json(force=True, silent=True) or {}).get("role") or "badscan"
    if role not in MODEL_ROLES:
        return jsonify({"error": f"Unknown role '{role}'"}), 400
    if role != "badscan":
        summary = retrain.dataset_summary(role)
        return jsonify({"role": role, "selection": None, "summary": summary})

    from .. import examples as ex_store

    ex_store.sync(role)
    all_examples = ex_store.for_training(role)
    chosen, report = retrain.select_badscan_examples(all_examples)
    from collections import Counter
    report["chosen_by_label"] = dict(Counter(e["label"] for e in chosen))
    report["chosen_total"] = len(chosen)
    report["store_total"] = len(all_examples)
    return jsonify({"role": role, "selection": report})


@bp.post("/start")
def start():
    """Kick off a retrain in the background."""
    data = request.get_json(force=True, silent=True) or {}
    role = (data.get("role") or "badscan").strip()
    if role not in MODEL_ROLES:
        return jsonify({"error": f"Unknown role '{role}'"}), 400
    if retrain.is_running():
        return jsonify({"error": "A training run is already in progress"}), 409

    settings = db.get_settings()
    try:
        epochs = int(data.get("epochs") or settings.get("retrain_epochs") or 50)
    except (TypeError, ValueError):
        return jsonify({"error": "epochs must be a number"}), 400
    if not 1 <= epochs <= 1000:
        return jsonify({"error": "epochs must be between 1 and 1000"}), 400

    auto = data.get("auto_activate")
    if auto is None:
        auto = settings.get("retrain_auto_activate", "1") == "1"

    dataset_id = data.get("dataset_id")
    retrain.train_async(role, epochs=epochs, auto_activate=bool(auto),
                        dataset_id=int(dataset_id) if dataset_id else None)
    return jsonify({"started": True, "role": role, "epochs": epochs,
                    "auto_activate": bool(auto)}), 202


@bp.get("/stream")
def stream():
    """
    Server-Sent Events feed of training progress.

    Emits a snapshot whenever it changes, a `done` event when the run
    finishes, and a comment line as a keepalive so proxies do not drop an
    idle connection.
    """
    def gen():
        last = None
        started = time.monotonic()
        was_running = retrain.is_running()
        while True:
            if time.monotonic() - started > IDLE_TIMEOUT:
                yield "event: timeout\ndata: {}\n\n"
                return

            running = retrain.is_running()
            snapshot = {
                "running": running,
                "progress": retrain.progress,
            }
            payload = json.dumps(snapshot, default=str)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            else:
                yield ": keepalive\n\n"

            if was_running and not running:
                # Include the finished run so the UI can show its outcome.
                final = {
                    "runs": db.list_training_runs(limit=1),
                    "active": {r: db.get_active_model(r) for r in MODEL_ROLES},
                }
                yield f"event: done\ndata: {json.dumps(final, default=str)}\n\n"
                return
            was_running = running or was_running
            time.sleep(POLL_SECONDS)

    return Response(gen(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",   # don't let a proxy buffer the stream
        "Connection": "keep-alive",
    })


@bp.get("/runs/<int:run_id>/history")
def run_history(run_id: int):
    """
    Per-epoch history for a finished run, so its curves can be re-opened
    later. The history is stored on the model the run produced.
    """
    import json as _json

    run = next((r for r in db.list_training_runs(limit=500) if r["id"] == run_id), None)
    if run is None:
        return jsonify({"error": "Run not found"}), 404
    history, metrics = [], {}
    try:
        metrics = _json.loads(run.get("model_metrics") or "{}")
        history = metrics.get("history") or []
    except (ValueError, TypeError):
        pass
    return jsonify({
        "run_id": run_id,
        "role": run["role"],
        "status": run["status"],
        "epochs": run["epochs"],
        "model_version": run.get("model_version"),
        "dataset_version": run.get("dataset_version"),
        "top1_accuracy": metrics.get("top1_accuracy"),
        "history": history,
    })
