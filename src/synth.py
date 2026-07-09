"""
Synthetic bad-scan generation.

The digitized manuscript images are essentially all good scans, so a
bad-scan classifier has no positive examples to learn from. This module
manufactures them: it degrades copies of clean images with the scanning
defects the augmentation module already models (smudges/dust, blur,
misorientation, obstructions) and writes them, plus untouched copies as
negatives, into SYNTHETIC_DIR/badscan/<label>/.

These persist and are folded into every bad-scan dataset snapshot, so the
synthetic positives keep contributing to training alongside real reviewer
verdicts.
"""

import random
from pathlib import Path

import cv2

from .config import SYNTHETIC_DIR
from .augmentation.blur import BlurAugmenter
from .augmentation.dust import DustAugmenter
from .augmentation.obstruction import ObstructionAugmenter
from .augmentation.orientation import OrientationAugmenter

BAD_DIR = SYNTHETIC_DIR / "badscan" / "bad_scan"
GOOD_DIR = SYNTHETIC_DIR / "badscan" / "no_bad_scan"


def _degrade(image, rng, augs):
    """Apply one random scanning defect to an image; return the degraded copy."""
    kind = rng.choice(["dust", "blur", "orientation", "obstruction"])
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


def generate(source_images: list[Path], per_class: int = 30, seed: int = 1337) -> dict:
    """
    Build a balanced synthetic bad-scan set from *source_images*.

    Half become degraded bad_scan positives, half stay as no_bad_scan
    negatives. Returns counts written.
    """
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
    imgs = imgs[: per_class * 2]
    bad_srcs = imgs[:per_class]
    good_srcs = imgs[per_class: per_class * 2]

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
