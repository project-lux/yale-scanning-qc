"""
Keep the scans the model got wrong.

The folder the app watches is a staging area: approved captures move on to
their permanent home within days and problem ones are re-captured, so
nothing in it can be relied on to still be there next week. The images
worth keeping long-term are the ones where an annotator overruled the
model — a flag they cleared, or a bad capture it missed. Those are the
training signal, and this module copies each one (as a reduced JPEG, never
the TIFF) into the app's own store together with everything known about it
at the time: where it came from, what the model said and with which
version, who reviewed it, when, and what they wrote.

One copy per image, named by content hash, shared by every role's example
row. A re-pass of the same file lands on the same copy; a re-capture under
the same filename is a different image and gets its own.
"""

import hashlib
import re
import shutil
from pathlib import Path

from . import db
from .config import IMPORT_JPEG_QUALITY, IMPORT_MAX_SIDE, KEPT_DIR

DISAGREEMENTS = ("false_positive", "false_negative")


# ---------------------------------------------------------------------------
# What a review means for each role
# ---------------------------------------------------------------------------

def label_for(role: str, rv: dict) -> tuple[str | None, str]:
    """
    (label, verdict) that a review gives for *role*, or (None, '') when it
    answers nothing for that role (a page-type-only review says nothing
    about bad scans, and vice versa).
    """
    if role == "badscan":
        v = rv.get("verdict") or rv.get("review_verdict") or ""
        if v in ("true_positive", "false_negative"):
            return "bad_scan", v
        if v in ("false_positive", "true_negative"):
            return "no_bad_scan", v
        return None, ""
    pt = rv.get("review_page_type") or ""
    return (pt or None), ""


def disagrees(role: str, rv: dict, label: str) -> bool:
    """True when the annotator's answer differs from what the model said."""
    if role == "badscan":
        return (rv.get("verdict") or rv.get("review_verdict")) in DISAGREEMENTS
    return (rv.get("page_type") or "") != label


def should_keep(role: str, rv: dict, label: str, settings: dict) -> bool:
    """
    The standing policy: keep what the model got wrong. Agreements are kept
    only when the `preserve_agreements` setting says so.
    """
    return disagrees(role, rv, label) or settings.get("preserve_agreements") == "1"


# ---------------------------------------------------------------------------
# Copying in
# ---------------------------------------------------------------------------

def source_for(rv: dict) -> Path | None:
    """
    The file to copy from: the cached web JPEG when there is one (it is
    what the model saw, and far smaller than the TIFF), else the original.
    None when neither is on disk any more.
    """
    for key in ("jpeg_path", "filepath", "image_path"):
        p = rv.get(key)
        if p and Path(p).is_file():
            return Path(p)
    return None


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("_")
    return (stem or "scan")[:80]


def kept_path(src: Path, original_name: str | None = None) -> tuple[Path, str]:
    """Where *src* would be kept, and its content hash."""
    sha = _sha1(src)
    return KEPT_DIR / f"{sha[:16]}_{_safe_stem(original_name or src.name)}.jpg", sha


def _copy_in(src: Path, dest: Path) -> Path | None:
    """
    Write *dest* from *src*: a straight copy when it is already a JPEG no
    larger than IMPORT_MAX_SIDE, otherwise a reduced JPEG. Idempotent.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(src) as im:
            if im.format == "JPEG" and max(im.size) <= IMPORT_MAX_SIDE:
                shutil.copy2(src, dest)
                return dest
            im = im.convert("RGB")
            im.thumbnail((IMPORT_MAX_SIDE, IMPORT_MAX_SIDE))
            tmp = dest.with_suffix(".part")
            im.save(tmp, "JPEG", quality=IMPORT_JPEG_QUALITY)
            tmp.replace(dest)
        return dest
    except Exception as e:
        print(f"[preserve] cannot copy {src.name}: {e}", flush=True)
        return None


def _relpath(filepath: str, project: dict | None) -> str:
    if project:
        try:
            return str(Path(filepath).relative_to(project["path"]))
        except ValueError:
            pass
    return Path(filepath).name


def snapshot(rv: dict, project: dict | None, src: Path, sha: str) -> dict:
    """
    Everything worth remembering about a scan at the moment it is kept.
    Accepts either a review-joined row (:func:`db.reviews_with_scans`) or a
    plain scan row from :func:`db.get_scan`.
    """
    filepath = rv.get("filepath") or ""
    return {
        "original_path": filepath,
        "relpath": _relpath(filepath, project),
        "directory": rv.get("directory") or str(Path(filepath).parent),
        "project_id": project["id"] if project else None,
        "project_name": project["name"] if project else None,
        "scan_id": rv.get("scan_id") or rv.get("id"),
        "batch": rv.get("batch_label"),
        "processed_in": rv.get("batch_started"),
        "badscan_model": rv.get("badscan_model_version"),
        "pagetype_model": rv.get("pagetype_model_version"),
        "model_bad_scan": rv.get("bad_scan"),
        "model_bad_scan_conf": rv.get("bad_scan_conf"),
        "model_page_type": rv.get("page_type"),
        "model_page_type_conf": rv.get("page_type_conf"),
        "model_page_type_source": rv.get("page_type_source"),
        "verdict": rv.get("verdict") or rv.get("review_verdict") or "",
        "review_page_type": rv.get("review_page_type"),
        "reviewer": rv.get("reviewer") or "",
        "reviewed_at": rv.get("reviewed_at"),
        "review_notes": rv.get("review_notes") or "",
        "file_size": rv.get("file_size"),
        "file_mtime": rv.get("file_mtime"),
        "kept_from": str(src),
        "kept_at": db.utcnow(),
        "sha1": sha,
    }


def keep(rv: dict, project: dict | None) -> dict | None:
    """
    Copy the scan behind *rv* into the store. Returns
    ``{"image_path", "meta", "sha1"}``, or None when the source file is
    already gone (nothing can be kept) or cannot be read.
    """
    src = source_for(rv)
    if src is None:
        print(f"[preserve] source gone, cannot keep {rv.get('filename') or rv.get('filepath')}",
              flush=True)
        return None
    dest, sha = kept_path(src, rv.get("filename") or Path(rv.get("filepath") or src).name)
    if _copy_in(src, dest) is None:
        return None
    return {"image_path": str(dest), "meta": snapshot(rv, project, src, sha), "sha1": sha}


def adopt(image_path: str, scan: dict | None, project: dict | None) -> dict | None:
    """
    Bring an already-labeled example whose image still sits outside the
    store (an older review pointing into a watched folder) into the kept
    store. Same return shape as :func:`keep`.
    """
    src = Path(image_path)
    if not src.is_file():
        return None
    rv = dict(scan or {})
    rv.setdefault("filepath", image_path)
    dest, sha = kept_path(src, Path(rv.get("filepath") or image_path).name)
    if _copy_in(src, dest) is None:
        return None
    return {"image_path": str(dest), "meta": snapshot(rv, project, src, sha), "sha1": sha}


def is_kept(image_path: str) -> bool:
    try:
        return Path(image_path).resolve().is_relative_to(KEPT_DIR.resolve())
    except (OSError, ValueError):
        return False
