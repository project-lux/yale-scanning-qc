"""
Client for the companion GPU server (see src/train_server.py).

Offloads the two heavy operations to the remote box when configured in
.env (TRAIN_SERVER_URL / SPARK_IP), and no-ops to local otherwise:

  training_enabled()  — fine-tune the fit step remotely
  inference_enabled() — run batch inference remotely (REMOTE_INFERENCE=1)
"""

import io
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import cv2
import requests

from .config import (
    REMOTE_INFERENCE,
    SPARK_IP,
    SPARK_PASSWORD,
    SPARK_PORT,
    SPARK_SSH_PORT,
    SPARK_USERNAME,
    TRAIN_SERVER_TOKEN,
    TRAIN_SERVER_URL,
)

POLL_INTERVAL = 3.0
POLL_MAX_SECONDS = 6 * 60 * 60  # safety ceiling for a single training job

# When username+password are configured the app opens (and maintains) its own
# SSH tunnel to the box, so it works on any machine with no manual `ssh -L`
# and no host alias. Otherwise it falls back to TRAIN_SERVER_URL.
_use_tunnel = bool(SPARK_IP and SPARK_USERNAME and SPARK_PASSWORD)

# One tunnel per remote port (train server, vLLM, …), each lazily opened.
_tunnels: dict[int, object] = {}
_tunnel_lock = threading.Lock()


def tunnel_port(remote_port: int) -> int:
    """
    Open (or reuse) an SSH tunnel to 127.0.0.1:<remote_port> on the box;
    return the local forwarded port. Auth goes through the SSH agent/keys
    first (like a plain `ssh`), with the configured password as fallback.
    """
    with _tunnel_lock:
        t = _tunnels.get(remote_port)
        if t is not None:
            try:
                if t.is_active:
                    return t.local_bind_port
            except Exception:
                pass
            try:
                t.stop()
            except Exception:
                pass
            _tunnels.pop(remote_port, None)

        import sshtunnel

        t = sshtunnel.SSHTunnelForwarder(
            (SPARK_IP, SPARK_SSH_PORT),
            ssh_username=SPARK_USERNAME,
            ssh_password=SPARK_PASSWORD,
            remote_bind_address=("127.0.0.1", remote_port),
            local_bind_address=("127.0.0.1", 0),
            set_keepalive=30.0,
        )
        t.start()
        _tunnels[remote_port] = t
        return t.local_bind_port


def _base_url() -> str:
    if _use_tunnel:
        return f"http://127.0.0.1:{tunnel_port(int(SPARK_PORT))}"
    return TRAIN_SERVER_URL


def close_tunnel() -> None:
    with _tunnel_lock:
        for t in _tunnels.values():
            try:
                t.stop()
            except Exception:
                pass
        _tunnels.clear()


def training_enabled() -> bool:
    """Config check only — does not open the tunnel."""
    return _use_tunnel or bool(TRAIN_SERVER_URL)


def inference_enabled() -> bool:
    return training_enabled() and REMOTE_INFERENCE


def _headers() -> dict:
    return {"Authorization": f"Bearer {TRAIN_SERVER_TOKEN}"} if TRAIN_SERVER_TOKEN else {}


def health(timeout: float = 10) -> dict:
    r = requests.get(f"{_base_url()}/health", headers=_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def reachable(timeout: float = 6) -> bool:
    """True if the server answers /health (opens the tunnel if needed)."""
    if not training_enabled():
        return False
    try:
        health(timeout=timeout)
        return True
    except Exception:
        return False


_status_cache = {"t": -1e9, "val": None}
_STATUS_TTL = 20.0


def status(refresh: bool = False) -> dict:
    """
    Cached remote-box status for the UI (avoids a health call every poll).
    Returns {enabled, use_tunnel, inference, reachable, gpu?, error?}.
    """
    if not training_enabled():
        return {"enabled": False, "reachable": False}
    now = time.monotonic()
    if not refresh and _status_cache["val"] and now - _status_cache["t"] < _STATUS_TTL:
        return _status_cache["val"]
    val = {"enabled": True, "use_tunnel": _use_tunnel, "inference": inference_enabled(),
           "host": SPARK_IP}
    try:
        h = health(timeout=5)
        val.update(reachable=True, gpu=h.get("gpu"), cuda=h.get("cuda"),
                   loaded_models=h.get("loaded_models", []))
    except Exception as e:
        val.update(reachable=False, error=str(e)[:140])
    _status_cache.update(t=now, val=val)
    return val


def _zip_dataset(ds_dir: Path) -> bytes:
    """Zip the train/ val/ classification layout with archive-root-relative paths."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for p in ds_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(ds_dir).as_posix())
    return buf.getvalue()


def train_remote(ds_dir: Path, base_weights: Path, params: dict,
                 progress_cb=None) -> tuple[Path, dict]:
    """
    Run one training job on the remote box.

    Uploads the dataset + base weights + params, polls to completion, and
    downloads the trained best.pt to a temp file. Returns (weights_path,
    metrics). Raises on remote error.
    """
    import json

    zbytes = _zip_dataset(ds_dir)
    with open(base_weights, "rb") as bf:
        files = {
            "dataset": ("dataset.zip", zbytes, "application/zip"),
            "base": ("base.pt", bf, "application/octet-stream"),
        }
        r = requests.post(f"{_base_url()}/train", headers=_headers(),
                          files=files, data={"params": json.dumps(params)}, timeout=600)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    deadline = time.monotonic() + POLL_MAX_SECONDS
    status = {}
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        jr = requests.get(f"{_base_url()}/jobs/{job_id}", headers=_headers(), timeout=30)
        jr.raise_for_status()
        status = jr.json()
        if progress_cb:
            progress_cb(status)
        if status.get("status") == "done":
            break
        if status.get("status") == "error":
            raise RuntimeError(f"remote training failed: {status.get('error')}")
    else:
        raise TimeoutError("remote training exceeded time limit")

    wr = requests.get(f"{_base_url()}/jobs/{job_id}/weights", headers=_headers(), timeout=300)
    wr.raise_for_status()
    tmp = Path(tempfile.mkdtemp(prefix="qc-remote-")) / "best.pt"
    tmp.write_bytes(wr.content)

    try:
        requests.delete(f"{_base_url()}/jobs/{job_id}", headers=_headers(), timeout=30)
    except Exception:
        pass
    return tmp, status.get("metrics", {})


def deploy_model(version: str, weights_path: Path) -> dict:
    """Upload a model version to the remote inference cache."""
    with open(weights_path, "rb") as wf:
        r = requests.post(f"{_base_url()}/models/{version}", headers=_headers(),
                          files={"weights": ("weights.pt", wf, "application/octet-stream")},
                          timeout=300)
    r.raise_for_status()
    return r.json()


def models_available() -> set[str]:
    """Model versions the remote already has (loaded or on disk)."""
    r = requests.get(f"{_base_url()}/models", headers=_headers(), timeout=30)
    r.raise_for_status()
    d = r.json()
    return set(d.get("loaded", [])) | set(d.get("on_disk", []))


def ensure_model(version: str, weights_path: Path) -> None:
    """Deploy a model version to the remote only if it isn't there already."""
    try:
        if version in models_available():
            return
    except Exception:
        pass
    deploy_model(version, weights_path)


def infer_batch(version: str, images: list, quality: int = 90) -> list[dict]:
    """
    Classify a list of BGR images on the remote box.

    Returns a list of {"class": str|None, "conf": float} aligned to input.
    """
    files = []
    for i, img in enumerate(images):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("failed to encode image for remote inference")
        files.append(("images", (f"{i}.jpg", buf.tobytes(), "image/jpeg")))
    r = requests.post(f"{_base_url()}/infer", headers=_headers(),
                      data={"version": version}, files=files, timeout=300)
    r.raise_for_status()
    return r.json()["predictions"]
