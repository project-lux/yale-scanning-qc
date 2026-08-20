"""
Synthetic bad-scan generation.

The digitized manuscript images are essentially all good scans, so a
bad-scan classifier has no positive examples to learn from. This module
manufactures them: it degrades copies of clean images with the scanning
defects the augmentation module already models (smudges/dust, blur,
misorientation, obstructions) plus two capture faults modeled here
(bad exposure, colour cast) and writes them, plus untouched copies as
negatives, into SYNTHETIC_DIR/badscan/<label>/.

Exposure, colour cast, framing and folded corners are the failure modes the
legacy reviewers recorded as missed ("image too dark/no light",
"discolorations", "cropping issue") and that the v008 classifier confidently
got wrong, so they are degraded here rather than left to the detection
augmenters.

These persist and are folded into every bad-scan dataset snapshot, so the
synthetic positives keep contributing to training alongside real reviewer
verdicts.
"""

import random
from pathlib import Path

import cv2
import numpy as np

from .config import SYNTHETIC_DIR
from .augmentation.blur import BlurAugmenter
from .augmentation.dust import DustAugmenter
from .augmentation.obstruction import ObstructionAugmenter
from .augmentation.orientation import OrientationAugmenter

BAD_DIR = SYNTHETIC_DIR / "badscan" / "bad_scan"
GOOD_DIR = SYNTHETIC_DIR / "badscan" / "no_bad_scan"


def _bad_exposure(image, rng):
    """
    Severe under- or over-exposure — a capture made with the lights off or
    blown out. Reviewers logged this as "image too dark/no light".
    """
    if rng.random() < 0.75:
        gain = rng.uniform(0.10, 0.28)   # very dark
    else:
        gain = rng.uniform(2.2, 3.6)     # blown highlights
    out = image.astype(np.float32) * gain
    # Crushed blacks / clipped whites also flatten local contrast.
    mid = out.mean()
    out = (out - mid) * rng.uniform(0.55, 0.9) + mid
    return np.clip(out, 0, 255).astype(np.uint8)


def _colour_cast(image, rng):
    """
    Strong channel imbalance — a white-balance failure. Reviewers logged
    this as "discolorations in images" (e.g. the heavy blue casts).
    """
    out = image.astype(np.float32)
    # One channel dominates, another is suppressed (BGR order).
    strong, weak = rng.sample([0, 1, 2], 2)
    out[:, :, strong] *= rng.uniform(1.35, 2.1)
    out[:, :, weak] *= rng.uniform(0.35, 0.7)
    if rng.random() < 0.4:
        # Banded cast: a horizontal strip of the frame shifts differently.
        h = out.shape[0]
        y0 = rng.randint(0, max(0, h - h // 6))
        y1 = y0 + max(1, h // rng.randint(6, 14))
        out[y0:y1] *= rng.uniform(1.2, 1.8)
    return np.clip(out, 0, 255).astype(np.uint8)


def _crop_error(image, rng):
    """
    Mis-framed capture: the frame cuts into the document so content runs off
    the edge with no margin, and the neighbouring page or platen intrudes.
    Reviewers logged this as "cropping issue" — the defect v008 missed most
    confidently.
    """
    h, w = image.shape[:2]
    # Crop hard into one or two sides, then restore the original size so the
    # page bleeds off the frame instead of sitting inside a margin.
    top = rng.randint(0, h // 6)
    bottom = h - rng.randint(0, h // 6)
    left = rng.randint(0, w // 6)
    right = w - rng.randint(0, w // 6)
    # Guarantee at least one side is clipped enough to matter.
    if (bottom - top) > h * 0.92 and (right - left) > w * 0.92:
        if rng.random() < 0.5:
            bottom = h - int(h * rng.uniform(0.10, 0.18))
        else:
            right = w - int(w * rng.uniform(0.10, 0.18))
    out = image[top:bottom, left:right]
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        # A dark band along one edge: the adjacent page / gutter creeping in.
        band = max(2, int(min(h, w) * rng.uniform(0.02, 0.06)))
        dark = int(rng.uniform(18, 55))
        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "top":
            out[:band] = dark
        elif side == "bottom":
            out[-band:] = dark
        elif side == "left":
            out[:, :band] = dark
        else:
            out[:, -band:] = dark
    return out


def _folded_corner(image, rng):
    """
    A corner of the document folded over (or a wedge of dark background
    lying across it) — the geometry behind several accepted bad captures.
    """
    out = image.copy()
    h, w = out.shape[:2]
    # Wedge depth as a fraction of the frame, big enough to read as a fault.
    fh = int(h * rng.uniform(0.10, 0.26))
    fw = int(w * rng.uniform(0.10, 0.30))
    corner = rng.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        tri = [(0, 0), (fw, 0), (0, fh)]
    elif corner == "tr":
        tri = [(w, 0), (w - fw, 0), (w, fh)]
    elif corner == "bl":
        tri = [(0, h), (fw, h), (0, h - fh)]
    else:
        tri = [(w, h), (w - fw, h), (w, h - fh)]
    # Tone it like the scanner background, not pure black.
    shade = int(rng.uniform(20, 70))
    cv2.fillPoly(out, [np.array(tri, dtype=np.int32)], (shade, shade, shade))
    return out


def _degrade(image, rng, augs):
    """Apply one random scanning defect to an image; return the degraded copy."""
    kind = rng.choice(["dust", "blur", "orientation", "obstruction",
                       "exposure", "colour_cast", "crop_error", "folded_corner"])
    if kind == "exposure":
        return kind, _bad_exposure(image, rng)
    if kind == "colour_cast":
        return kind, _colour_cast(image, rng)
    if kind == "crop_error":
        return kind, _crop_error(image, rng)
    if kind == "folded_corner":
        return kind, _folded_corner(image, rng)
    if kind == "dust":
        out, _ = augs["dust"].augment(image, severity=rng.choice(["medium", "heavy"]))
    elif kind == "blur":
        out, _ = augs["blur"].augment(image, severity=rng.choice(["medium", "heavy"]))
    elif kind == "orientation":
        out, _ = augs["orientation"].augment(
            image, target_class=rng.choice(["rotate_90_cw", "rotate_180", "rotate_90_ccw"]))
    else:
        out, _ = augs["obstruction"].augment(image)
    return kind, out


def generate(source_images: list[Path], per_class: int = 30, seed: int = 1337,
             n_good: int | None = None) -> dict:
    """
    Build a synthetic bad-scan set from *source_images*.

    *per_class* images become degraded bad_scan positives and *n_good*
    (default: *per_class*, for a balanced set) stay untouched as
    no_bad_scan negatives, drawn from a disjoint slice. Pass n_good=0 when
    the negatives already come from a real curated set, so this only tops
    up the positives. Returns counts written.
    """
    if n_good is None:
        n_good = per_class
    rng = random.Random(seed)
    augs = {
        "dust": DustAugmenter(seed),
        "blur": BlurAugmenter(seed),
        "orientation": OrientationAugmenter(seed),
        "obstruction": ObstructionAugmenter(seed),
    }
    BAD_DIR.mkdir(parents=True, exist_ok=True)
    GOOD_DIR.mkdir(parents=True, exist_ok=True)

    imgs = list(source_images)
    rng.shuffle(imgs)
    imgs = imgs[: per_class + n_good]
    bad_srcs = imgs[:per_class]
    good_srcs = imgs[per_class: per_class + n_good]

    n_bad = 0
    for i, p in enumerate(bad_srcs):
        image = cv2.imread(str(p))
        if image is None:
            continue
        kind, degraded = _degrade(image, rng, augs)
        cv2.imwrite(str(BAD_DIR / f"{i:04d}_{kind}_{p.stem}.jpg"), degraded)
        n_bad += 1

    n_good = 0
    for i, p in enumerate(good_srcs):
        image = cv2.imread(str(p))
        if image is None:
            continue
        cv2.imwrite(str(GOOD_DIR / f"{i:04d}_{p.stem}.jpg"), image)
        n_good += 1

    return {"bad_scan": n_bad, "no_bad_scan": n_good}
