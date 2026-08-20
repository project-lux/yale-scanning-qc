"""
Cached grid thumbnails.

Source images are archival derivatives up to 2500px and a grid can show
hundreds at once, so every tile gets a small cached JPEG keyed by the
caller's own id namespace.
"""

from pathlib import Path

from flask import send_file

from ..config import DATA_DIR

THUMB_DIR = DATA_DIR / "thumbs"
THUMB_SIDE = 320


def serve(kind: str, key: int | str, src: Path, side: int = THUMB_SIDE):
    """
    Send a cached thumbnail of *src*, regenerating it when the source is
    newer. Falls back to the original on any decode failure — a thumbnail
    is a convenience, never a reason to show nothing.
    """
    out_dir = THUMB_DIR / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb = out_dir / f"{key}_{side}.jpg"
    try:
        if not thumb.exists() or thumb.stat().st_mtime < src.stat().st_mtime:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((side, side))
                im.save(thumb, "JPEG", quality=82)
        return send_file(thumb, mimetype="image/jpeg")
    except Exception:
        return send_file(src)


def invalidate(kind: str, key: int | str) -> None:
    """Drop cached thumbnails for one item (after its image changes)."""
    out_dir = THUMB_DIR / kind
    if not out_dir.is_dir():
        return
    for p in out_dir.glob(f"{key}_*.jpg"):
        p.unlink(missing_ok=True)
