"""
Bring training images in from the DSCA share.

Two jobs, both of which only ever copy a reduced JPEG (IMPORT_MAX_SIDE):
the archival TIFFs are far larger than the UI or the trainer needs, and the
share is slow, so the full files are never pulled across.

  import_bad_files()      the explicitly-bad captures under "1) Bad files"
  import_project_images() presumed-clean pages from a project, the pool the
                          dataset builder samples negatives from

Both are resumable: a derivative that already exists is not re-read, and
rows are inserted only when absent, so a corrected label is never clobbered.
"""

import random
from pathlib import Path

from . import db
from .config import IMAGES_DIR, IMPORT_JPEG_QUALITY, IMPORT_MAX_SIDE

SHARE_ROOT = Path("/Volumes/DSCA_Digital_Studio-CC1702-LIBITS/AI Project/AI Project")
BAD_FILES_DIR = SHARE_ROOT / "1) Bad files"

IMAGE_SUFFIXES = (".tif", ".tiff", ".jpg", ".jpeg", ".png")


def derive(src: Path, dest: Path, max_side: int = IMPORT_MAX_SIDE) -> Path | None:
    """
    Write a reduced JPEG copy of *src* at *dest*. Returns dest, or None if
    the image could not be read. Skips work when dest is already current.
    """
    try:
        if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
            return dest
    except OSError:
        return None

    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            im.save(dest, "JPEG", quality=IMPORT_JPEG_QUALITY)
        return dest
    except Exception as e:
        print(f"[import] cannot read {src.name}: {e}", flush=True)
        return None


def _flat_name(path: Path, root: Path) -> str:
    """A collision-free filename encoding the path relative to *root*."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    return "__".join(rel.parts[:-1] + (rel.stem,)).replace(" ", "_") + ".jpg"


# ---------------------------------------------------------------------------
# Explicitly-bad captures
# ---------------------------------------------------------------------------

def _cleared_by_reviewer(qc_dir: Path) -> set[str]:
    """
    Page stems a reviewer explicitly marked "ignore" (false-positive), i.e.
    the model flagged them but a human said they were fine. These must not
    be imported as bad, even from a folder named "Bad Captures".
    """
    fp = qc_dir / "false-positive"
    if not fp.is_dir():
        return set()
    return {p.stem for p in fp.glob("*.json")}


def collect_bad_files() -> dict[str, list[Path]]:
    """
    The images under "1) Bad files" that are clearly delineated as bad,
    grouped by how we know. Nothing is included merely because it sits in
    that folder: the two large ru_151 item folders are ordinary clean
    volumes and are skipped entirely.
    """
    groups: dict[str, list[Path]] = {
        "reviewer_confirmed": [],   # qc/true-positive  — flag accepted
        "reviewer_missed": [],      # qc/false-negative — human caught a miss
        "loose_bad_captures": [],   # "Singular images" — filed as bad on its own
        "bad_captures_folder": [],  # folder named *Bad_Captures*, not cleared
    }
    if not BAD_FILES_DIR.is_dir():
        return groups

    for item in sorted(BAD_FILES_DIR.iterdir()):
        if item.is_file():
            continue
        name = item.name

        if name == "Singular images":
            groups["loose_bad_captures"] += [
                p for p in sorted(item.iterdir())
                if p.suffix.lower() in IMAGE_SUFFIXES
            ]
            continue

        qc = item / "qc"
        if qc.is_dir():
            for sub, key in (("true-positive", "reviewer_confirmed"),
                             ("false-negative", "reviewer_missed")):
                d = qc / sub
                if d.is_dir():
                    groups[key] += sorted(d.glob("*.jpg"))

        # A folder whose name says the captures are bad, minus any page a
        # reviewer cleared and any page already counted above.
        if "bad_capture" in name.lower():
            cleared = _cleared_by_reviewer(qc)
            counted = {p.stem for p in groups["reviewer_confirmed"] + groups["reviewer_missed"]}
            groups["bad_captures_folder"] += [
                p for p in sorted(item.iterdir())
                if p.suffix.lower() in IMAGE_SUFFIXES
                and p.stem not in cleared and p.stem not in counted
            ]
    return groups


def import_bad_files(role: str = "badscan") -> dict:
    """Import every clearly-bad capture as a bad_scan training example."""
    groups = collect_bad_files()
    out_dir = IMAGES_DIR / role / "bad"
    added, skipped, counts = 0, 0, {}

    for group, paths in groups.items():
        n = 0
        for src in paths:
            dest = out_dir / _flat_name(src, BAD_FILES_DIR)
            if derive(src, dest) is None:
                skipped += 1
                continue
            ex_id = db.add_training_example(
                role=role, label="bad_scan", source="curated",
                image_path=str(dest), origin_ref=str(src),
                notes=f"explicit bad capture ({group.replace('_', ' ')})",
            )
            if ex_id is not None:
                added += 1
                n += 1
        counts[group] = n
        print(f"[import] {group}: {n} imported ({len(paths)} candidates)", flush=True)

    return {"added": added, "skipped": skipped, "groups": counts}


# ---------------------------------------------------------------------------
# Presumed-clean project pages (the negative pool)
# ---------------------------------------------------------------------------

def project_images(project_path: Path) -> list[Path]:
    """Top-level images of an item folder (skips qc/ and nested oddities)."""
    if not project_path.is_dir():
        return []
    return sorted(p for p in project_path.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def import_project_images(project_id: int, cap: int = 60,
                          role: str = "badscan", seed: int = 1337) -> dict:
    """
    Import up to *cap* pages from a project as presumed-clean negatives.

    These are labeled no_bad_scan because archival captures are
    overwhelmingly fine, and they are tagged source='project' so they stay
    distinguishable from a human-verified negative — the dataset builder
    samples from them rather than taking them all.
    """
    project = db.get_project(project_id)
    if project is None:
        return {"error": f"No project {project_id}"}

    root = Path(project["path"])
    files = project_images(root)
    if len(files) > cap:
        files = random.Random(seed).sample(files, cap)
        files.sort()

    out_dir = IMAGES_DIR / role / "project"
    added, skipped = 0, 0
    for src in files:
        dest = out_dir / _flat_name(src, root.parent)
        if derive(src, dest) is None:
            skipped += 1
            continue
        if db.add_training_example(
            role=role, label="no_bad_scan", source="project",
            image_path=str(dest), project_id=project_id, origin_ref=str(src),
            notes="presumed-clean project page (negative pool)",
        ) is not None:
            added += 1
    print(f"[import] {project['name']}: {added} negatives "
          f"({len(files)} sampled of {len(project_images(root))})", flush=True)
    return {"added": added, "skipped": skipped, "project": project["name"]}
