"""
Local OCR through kraken with a PP-OCRv6 recognition model.

Unlike the LLM transcription backends this runs entirely on the QC host:
kraken's default baseline segmenter finds the text lines and the
recognition model (small-models-for-glam/kraken-ppocrv6-medium, a
44-language CTC line recognizer trained on historical, contemporary and
born-digital documents) reads each one. Works on handwriting and print.

The model pair is loaded once and cached; changing kraken_model_path in
Settings triggers a reload on the next call. Get the default weights with:

    hf download small-models-for-glam/kraken-ppocrv6-medium \
        medium.safetensors --local-dir models/kraken-ppocrv6-medium
"""

import logging
import threading
import warnings
from pathlib import Path

from .config import MODELS_DIR

DEFAULT_KRAKEN_MODEL = MODELS_DIR / "kraken-ppocrv6-medium" / "medium.safetensors"

# One engine at a time; kraken/torch inference is not re-entrant and the
# pipeline already serializes work, so a plain lock is enough.
_lock = threading.Lock()
_engine: dict | None = None  # {"path": str, "seg": model, "rec": model}


def resolve_model_path(settings: dict) -> Path:
    raw = (settings.get("kraken_model_path") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_KRAKEN_MODEL


def _accelerator(settings: dict) -> str:
    """Requested compute device, downgraded to 'auto' when unavailable."""
    device = (settings.get("kraken_device") or "mps").strip().lower()
    if device == "mps":
        import torch
        if not torch.backends.mps.is_available():
            return "auto"
    return device or "auto"


def _load(model_path: Path, accelerator: str) -> dict:
    # kraken pulls in coremltools/lightning, which are noisy at import time.
    warnings.filterwarnings("ignore", module="coremltools")
    logging.getLogger("kraken").setLevel(logging.ERROR)

    from importlib import resources

    from kraken.configs import (
        RecognitionInferenceConfig,
        SegmentationInferenceConfig,
    )
    from kraken.models import load_models

    seg_path = str(resources.files("kraken").joinpath("blla.mlmodel"))
    seg = load_models(seg_path, tasks=["segmentation"])[0]
    rec = load_models(str(model_path), tasks=["recognition"])[0]
    seg.prepare_for_inference(SegmentationInferenceConfig(accelerator=accelerator))
    # num_line_workers=0 keeps line extraction in-process; the default
    # spawns a torch.multiprocessing pool that misbehaves under the
    # pipeline's worker threads on macOS.
    rec.prepare_for_inference(RecognitionInferenceConfig(
        accelerator=accelerator, num_line_workers=0))
    return {"path": str(model_path), "accelerator": accelerator,
            "seg": seg, "rec": rec}


def _points(pts) -> list[list[int]] | None:
    """Round a kraken point sequence to plain [[x, y], …] for JSON."""
    if not pts:
        return None
    return [[int(round(x)), int(round(y))] for x, y in pts]


def transcribe_kraken(jpeg_path: Path, settings: dict) -> tuple[str, dict]:
    """
    OCR one page image.

    Returns (text, geometry): *text* is the recognized lines joined with
    newlines; *geometry* is a JSON-ready dict with the image size and one
    entry per non-empty line (text, bounding polygon, baseline, mean
    confidence), in the same order as the text rows — it powers the
    line-overlay viewer in the UI.

    Raises if the model file is missing or the image is unreadable, so the
    pipeline records the error on the scan like any other backend failure.
    """
    model_path = resolve_model_path(settings)
    if not model_path.exists():
        raise FileNotFoundError(
            f"kraken model not found: {model_path} — download it or fix "
            "kraken_model_path in Settings")

    from PIL import Image

    accelerator = _accelerator(settings)

    global _engine
    with _lock:
        if (_engine is None or _engine["path"] != str(model_path)
                or _engine["accelerator"] != accelerator):
            _engine = _load(model_path, accelerator)

        im = Image.open(jpeg_path)
        seg = _engine["seg"].predict(im=im)
        preds = _engine["rec"].predict(im=im, segmentation=seg)

        lines = []
        for p in preds:
            text = (p.prediction or "").strip()
            if not text:
                continue
            confs = list(p.confidences or [])
            lines.append({
                "text": text,
                "polygon": _points(p.boundary),
                "baseline": _points(p.baseline),
                "confidence": round(sum(confs) / len(confs), 4) if confs else None,
            })

    geometry = {"w": im.width, "h": im.height, "lines": lines}
    return "\n".join(ln["text"] for ln in lines), geometry
