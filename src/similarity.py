"""
Visual similarity search: "show me images that look like this one".

Each image gets one embedding from a small self-supervised vision model
(DINOv2-small, 384 dims). DINOv2 features capture layout, texture and
tone rather than meaning, which is what matters for scans: pages with the
same blur, skew, obstruction or page type land close together.

Search is brute-force cosine similarity over an in-memory float32 matrix.
At 100k images that is ~150 MB and a single matrix-vector product of a
few milliseconds, so no approximate index is needed. Embeddings are kept
in SQLite (``image_embeddings``) so they survive restarts, and a single
background worker embeds whatever is not indexed yet.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import db

MODEL_NAME = "facebook/dinov2-small"
SIDE = 224
BATCH = 32
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model_lock = threading.Lock()
_model = None
_device = None

_index_lock = threading.Lock()
_paths: list[str] = []
_pos: dict[str, int] = {}
_buf = np.zeros((0, 0), dtype=np.float32)   # capacity-doubling row buffer
_n = 0                                        # rows in use
_loaded = False

_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
_rerun = threading.Event()
_failed: set[str] = set()
progress: dict = {"running": False, "done": 0, "todo": 0}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_model():
    global _model, _device
    if _model is None:
        import torch
        from transformers import AutoModel

        _device = ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
        _model = AutoModel.from_pretrained(MODEL_NAME).to(_device).eval()
    return _model


def _read(path: str) -> np.ndarray | None:
    """Decode one image to a normalised 224x224 RGB array, or None."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as im:
            # JPEG draft mode decodes at 1/2–1/8 scale: most of the speed-up
            # on 2500px archival derivatives.
            im.draft("RGB", (SIDE * 2, SIDE * 2))
            # Squash rather than crop: page edges (cut-off text, fingers,
            # clamps) are exactly the defects worth matching.
            im = im.convert("RGB").resize((SIDE, SIDE), Image.BILINEAR)
            a = np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None
    return ((a - _MEAN) / _STD).transpose(2, 0, 1)


def embed_paths(paths: list[str], pool: ThreadPoolExecutor | None = None
                ) -> dict[str, np.ndarray]:
    """Embed images (decoded in parallel). Unreadable files are skipped."""
    import torch

    if pool is None:
        with ThreadPoolExecutor(max_workers=8) as p:
            return embed_paths(paths, p)
    arrays = list(pool.map(_read, paths))
    ok = [(p, a) for p, a in zip(paths, arrays) if a is not None]
    if not ok:
        return {}
    with _model_lock:
        model = _load_model()
        x = torch.from_numpy(np.stack([a for _, a in ok])).to(_device)
        with torch.inference_mode():
            v = model(pixel_values=x).pooler_output.float().cpu().numpy()
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return {p: v[i] for i, (p, _) in enumerate(ok)}


# ---------------------------------------------------------------------------
# In-memory index
# ---------------------------------------------------------------------------

def _ensure_loaded() -> None:
    global _loaded, _buf, _n
    if _loaded:
        return
    with _index_lock:
        if _loaded:
            return
        paths, vecs = [], []
        for row in db.load_embeddings(MODEL_NAME):
            paths.append(row["path"])
            vecs.append(np.frombuffer(row["vec"], dtype=np.float16))
        _paths[:] = paths
        _pos.clear()
        _pos.update({p: i for i, p in enumerate(paths)})
        _buf = (np.stack(vecs).astype(np.float32) if vecs
                else np.zeros((0, 0), dtype=np.float32))
        _n = len(vecs)
        _loaded = True


def _add(found: dict[str, np.ndarray]) -> None:
    """Persist new embeddings and append them to the in-memory matrix."""
    global _buf, _n
    if not found:
        return
    db.save_embeddings(MODEL_NAME, [(p, v.astype(np.float16).tobytes())
                                    for p, v in found.items()])
    with _index_lock:
        new = [p for p in found if p not in _pos]
        for p in found:
            if p in _pos:
                _buf[_pos[p]] = found[p]
        if new:
            dim = len(found[new[0]])
            need = _n + len(new)
            if _buf.shape[0] < need or _buf.shape[1] != dim:
                grown = np.zeros((max(need, 2 * _buf.shape[0], 1024), dim), dtype=np.float32)
                if _n:
                    grown[:_n] = _buf[:_n]
                _buf = grown
            for p in new:
                _buf[_n] = found[p]
                _pos[p] = _n
                _paths.append(p)
                _n += 1


def indexed_count() -> int:
    _ensure_loaded()
    return len(_paths)


def vector_for(path: str) -> np.ndarray | None:
    """The stored embedding for *path*, embedding it now if it is missing."""
    _ensure_loaded()
    i = _pos.get(path)
    if i is not None:
        return _buf[i].copy()
    found = embed_paths([path])
    _add(found)
    return found.get(path)


def similar(path: str, scope: dict[str, int] | None = None, limit: int = 200,
            min_score: float = 0.0) -> list[tuple[int | str, float]]:
    """
    Images most similar to *path*, best first.

    *scope* maps candidate image paths to the caller's ids (scan or example
    ids); results are restricted to it and returned as (id, score). Without
    a scope, results are (path, score) over everything indexed.
    """
    q = vector_for(path)
    if q is None:
        return []
    with _index_lock:
        if _n == 0:
            return []
        # Scoring everything (one mat-vec) is cheaper than slicing the matrix.
        scores = _buf[:_n] @ q
        paths = _paths  # appends only ever grow it
    rows = None
    if scope is not None:
        rows = np.fromiter((_pos[p] for p in scope if p in _pos), dtype=np.int64)
        if not rows.size:
            return []
        scores = scores[rows]
    k = min(limit, scores.size)
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    out = []
    for j in top:
        s = float(scores[j])
        if s < min_score:
            break
        p = paths[rows[j] if rows is not None else j]
        out.append((scope[p] if scope is not None else p, s))
    return out


# ---------------------------------------------------------------------------
# Background indexing
# ---------------------------------------------------------------------------

def _missing() -> list[str]:
    _ensure_loaded()
    return [p for p in dict.fromkeys(db.embeddable_image_paths())
            if p not in _pos and p not in _failed]


def _run() -> None:
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                _rerun.clear()
                todo = _missing()
                progress.update(running=bool(todo), done=0, todo=len(todo))
                for i in range(0, len(todo), BATCH):
                    chunk = todo[i:i + BATCH]
                    exists = [p for p in chunk if Path(p).is_file()]
                    found = embed_paths(exists, pool)
                    _failed.update(p for p in chunk if p not in found)
                    _add(found)
                    progress["done"] = min(len(todo), i + BATCH)
                if not _rerun.is_set():
                    break
    except Exception as e:  # never take the app down over the index
        print(f"[similarity] indexing stopped: {e}")
    finally:
        progress["running"] = False


def index_missing_async(delay: float = 0.0) -> None:
    """Embed every not-yet-indexed image in a background thread."""
    global _worker

    def start():
        global _worker
        if delay:
            time.sleep(delay)
        with _worker_lock:
            if _worker is not None and _worker.is_alive():
                _rerun.set()  # pick up the new images when this run ends
                return
            _worker = threading.Thread(target=_run, daemon=True, name="similarity-index")
            _worker.start()

    threading.Thread(target=start, daemon=True).start()


def status() -> dict:
    return {
        "model": MODEL_NAME,
        "indexed": indexed_count(),
        "running": progress["running"],
        "done": progress["done"],
        "todo": progress["todo"],
    }
