# Yale Scanning QC — companion train/infer server

Runs on a GPU box (the DGX Spark) and does the heavy YOLO work the QC app
offloads to it: fine-tuning models and batch inference over passes.

It is **self-contained** — a single `train_server.py` plus `flask`. It does
NOT need the QC database, the registry, or the rest of the project.

## Deployment on the DGX Spark (as user `qc_user`)

Code lives at `~/apps/yolo-training/` on the box and runs as a persistent
systemd **user** service (`qc-trainer`), reached from the QC app over an
SSH tunnel the app opens itself from `SPARK_*` in `.env`.

```
/home/qc_user/apps/yolo-training/
├── train_server.py     # the service (copied from src/train_server.py)
├── pyproject.toml, run.sh, uv.lock, .venv/   # uv-managed env (torch+CUDA, ultralytics)
├── server.log          # service stdout/stderr (tail this to monitor)
└── work/               # jobs/ and models/ at runtime
```

## Run (systemd user service, survives reboot/logout)

```bash
# one-time
loginctl enable-linger "$USER"
# unit at ~/.config/systemd/user/qc-trainer.service sets QC_TRAIN_TOKEN, port 8050
export XDG_RUNTIME_DIR=/run/user/$(id -u)
systemctl --user enable --now qc-trainer

# monitor
tail -f ~/apps/yolo-training/server.log
systemctl --user status qc-trainer
```

Or run in the foreground for a quick test:

```bash
cd ~/apps/yolo-training
QC_TRAIN_TOKEN=<shared-secret> QC_TRAIN_PORT=8050 ./run.sh
```

Environment variables:
- `QC_TRAIN_TOKEN`   — bearer token; the QC app must send the same one. If unset, auth is OFF.
- `QC_TRAIN_WORKDIR` — where jobs/models are stored (default `./qc-trainer-work`).
- `QC_TRAIN_DEVICE`  — `""` (auto), `"0"`, `"cpu"`, … passed to ultralytics.

## HTTP contract

- `GET  /health` — torch/cuda/ultralytics info, loaded models.
- `POST /train` (multipart: `dataset` zip, `base` weights, `params` JSON) → `{job_id}`.
- `GET  /jobs/<id>` → `{status, epoch, epochs, metrics, error}`.
- `GET  /jobs/<id>/weights` → `best.pt` when `status=done`.
- `DELETE /jobs/<id>` — clean up job artifacts.
- `POST /models/<version>` (multipart: `weights`) — load a model for inference.
- `POST /infer` (multipart: `version` + `images[]`) → `{predictions:[{class,conf}]}`.

A single GPU lock serializes training and inference.
