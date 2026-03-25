"""
Directory scanner and file management for the QC workflow.

Handles TIFF discovery, JPEG cache generation, and the accept/ignore
workflow for flagged scans (true-positive / false-positive bookkeeping).
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .config import (
    JPEG_MAX_SIDE,
    JPEG_QUALITY,
    QC_CACHE_DIR,
    QC_FAILED_TIFFS_DIR,
    QC_FALSE_POSITIVE_DIR,
    QC_SUBDIR,
    QC_TRUE_POSITIVE_DIR,
)

TIFF_EXTENSIONS = {".tif", ".tiff"}
STATE_FILENAME = "state.json"


def _qc_root(watch_dir: Path) -> Path:
    return watch_dir / QC_SUBDIR


def _cache_dir(watch_dir: Path) -> Path:
    return _qc_root(watch_dir) / QC_CACHE_DIR


def _failed_dir(watch_dir: Path) -> Path:
    return _qc_root(watch_dir) / QC_FAILED_TIFFS_DIR


def _tp_dir(watch_dir: Path) -> Path:
    return _qc_root(watch_dir) / QC_TRUE_POSITIVE_DIR


def _fp_dir(watch_dir: Path) -> Path:
    return _qc_root(watch_dir) / QC_FALSE_POSITIVE_DIR


def ensure_qc_dirs(watch_dir: Path) -> None:
    """Create all QC subdirectories under the watched directory."""
    for d in [_cache_dir(watch_dir), _failed_dir(watch_dir), _tp_dir(watch_dir), _fp_dir(watch_dir)]:
        d.mkdir(parents=True, exist_ok=True)


def scan_tiffs(watch_dir: Path) -> list[str]:
    """Return TIFF filenames sorted by modification time, newest first."""
    entries = []
    for entry in os.scandir(watch_dir):
        if entry.is_file() and Path(entry.name).suffix.lower() in TIFF_EXTENSIONS:
            entries.append((entry.name, entry.stat().st_mtime))
    entries.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in entries]


def jpeg_cache_path(watch_dir: Path, tiff_name: str) -> Path:
    """Return the path where the cached JPEG should live."""
    base = Path(tiff_name).stem
    return _cache_dir(watch_dir) / f"{base}.jpg"


def convert_tiff_to_jpeg(tiff_path: Path, output_path: Path) -> Path:
    """
    Convert a TIFF to a JPEG suitable for web preview.

    Handles RGBA/P/LA mode conversion and resizes if either dimension
    exceeds JPEG_MAX_SIDE.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(tiff_path) as img:
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if w > JPEG_MAX_SIDE or h > JPEG_MAX_SIDE:
            scale = JPEG_MAX_SIDE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)

        img.save(output_path, "JPEG", quality=JPEG_QUALITY)

    return output_path


def ensure_jpeg_cached(watch_dir: Path, tiff_name: str) -> Path:
    """Convert the TIFF to JPEG if not already cached. Returns JPEG path."""
    jp = jpeg_cache_path(watch_dir, tiff_name)
    tiff_path = watch_dir / tiff_name

    if jp.exists():
        tiff_mtime = tiff_path.stat().st_mtime
        jpeg_mtime = jp.stat().st_mtime
        if jpeg_mtime >= tiff_mtime:
            return jp

    return convert_tiff_to_jpeg(tiff_path, jp)


def get_file_status(watch_dir: Path, tiff_name: str) -> str:
    """
    Derive a file's status from filesystem artifacts.

    Returns one of: pending, clean, flagged, accepted, ignored
    """
    base = Path(tiff_name).stem

    if (_tp_dir(watch_dir) / f"{base}.json").exists():
        return "accepted"
    if (_fp_dir(watch_dir) / f"{base}.json").exists():
        return "ignored"

    return "pending"


# ---------------------------------------------------------------------------
# Persistent state (state.json inside .qc/)
# ---------------------------------------------------------------------------

def _state_path(watch_dir: Path) -> Path:
    return _qc_root(watch_dir) / STATE_FILENAME


def load_state(watch_dir: Path) -> dict[str, dict]:
    """
    Load persisted file states from ``.qc/state.json``.

    Returns a dict keyed by filename, each value being
    ``{"status": ..., "results": ...}``.  The ``annotated_image``
    field (large base64 blob) is stripped on save to keep the file
    small; it will be regenerated on next analysis if needed.
    """
    sp = _state_path(watch_dir)
    if not sp.exists():
        return {}
    try:
        with open(sp) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(watch_dir: Path, file_states: dict[str, dict]) -> None:
    """
    Persist the in-memory file-state map to ``.qc/state.json``.

    Strips ``annotated_image`` from results before writing to avoid
    bloating the file with base64 image data.
    """
    ensure_qc_dirs(watch_dir)
    sp = _state_path(watch_dir)

    serializable = {}
    for name, state in file_states.items():
        entry = {"status": state.get("status", "pending")}
        results = state.get("results")
        if results:
            entry["results"] = {
                k: v for k, v in results.items() if k != "annotated_image"
            }
        else:
            entry["results"] = None
        serializable[name] = entry

    tmp = sp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    tmp.replace(sp)


def save_metadata(
    watch_dir: Path,
    tiff_name: str,
    results: dict,
    decision: str,
    models_used: list[str],
    image_size: tuple[int, int] | None = None,
) -> Path:
    """
    Write a metadata JSON file for training-data collection.

    *decision* is either ``"accept"`` or ``"ignore"``.
    Returns the path to the written JSON file.
    """
    base = Path(tiff_name).stem

    if decision == "accept":
        dest_dir = _tp_dir(watch_dir)
    else:
        dest_dir = _fp_dir(watch_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    flagged_by = [
        cat for cat, res in results.items()
        if isinstance(res, dict) and res.get("status") == "fail"
    ]

    meta = {
        "filename": tiff_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_used": models_used,
        "results": {
            k: v for k, v in results.items()
            if isinstance(v, dict) and "status" in v
        },
        "overall": results.get("overall", "unknown"),
        "flagged_by": flagged_by,
        "decision": decision,
    }
    if image_size:
        meta["image_size"] = list(image_size)

    json_path = dest_dir / f"{base}.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return json_path


def accept_file(watch_dir: Path, tiff_name: str, results: dict, models_used: list[str]) -> None:
    """
    Handle an *Accept* decision (true positive).

    1. Save metadata + JPEG to true-positive/
    2. Move the original TIF to failed-tiffs/
    """
    base = Path(tiff_name).stem
    tiff_path = watch_dir / tiff_name

    tif_size = None
    if tiff_path.exists():
        with Image.open(tiff_path) as img:
            tif_size = img.size

    save_metadata(watch_dir, tiff_name, results, "accept", models_used, tif_size)

    cached_jpeg = jpeg_cache_path(watch_dir, tiff_name)
    tp_jpeg = _tp_dir(watch_dir) / f"{base}.jpg"
    if cached_jpeg.exists():
        shutil.copy2(cached_jpeg, tp_jpeg)

    failed_dest = _failed_dir(watch_dir) / tiff_name
    _failed_dir(watch_dir).mkdir(parents=True, exist_ok=True)
    if tiff_path.exists():
        shutil.move(str(tiff_path), str(failed_dest))


def ignore_file(watch_dir: Path, tiff_name: str, results: dict, models_used: list[str]) -> None:
    """
    Handle an *Ignore* decision (false positive).

    1. Save metadata + JPEG to false-positive/
    2. Keep the original TIF in place.
    """
    base = Path(tiff_name).stem
    tiff_path = watch_dir / tiff_name

    tif_size = None
    if tiff_path.exists():
        with Image.open(tiff_path) as img:
            tif_size = img.size

    save_metadata(watch_dir, tiff_name, results, "ignore", models_used, tif_size)

    cached_jpeg = jpeg_cache_path(watch_dir, tiff_name)
    fp_jpeg = _fp_dir(watch_dir) / f"{base}.jpg"
    if cached_jpeg.exists():
        shutil.copy2(cached_jpeg, fp_jpeg)
