"""
Batch QC pipeline — replaces the real-time watcher.

A batch (a "pass") sweeps the monitored directories for image files that
have never been processed, runs the active bad-scan + pagetype models on
each, transcribes handwriting pages through the configured endpoint, and
records everything in the database. Each pass records the model versions
it ran with, so passes can be repeated after retraining and compared.
"""

import json
import os
import threading
import traceback
from pathlib import Path

import cv2

from . import db, inference, transcribe
from .config import PAGE_TYPES, SCAN_EXTENSIONS, all_page_types
from .scanner import convert_tiff_to_jpeg

# Only one batch runs at a time.
_batch_lock = threading.Lock()
_cancel = threading.Event()

# Live progress for the UI ({} when idle).
progress: dict = {}


def is_running() -> bool:
    return _batch_lock.locked()


def cancel() -> bool:
    """Request cancellation of the running batch. Returns True if one was running."""
    if not _batch_lock.locked():
        return False
    _cancel.set()
    return True


def discover_files(directory: Path) -> list[Path]:
    """
    Image files anywhere under *directory*, recursively.

    Skips hidden directories and the qc/ subtree (JPEG cache and legacy
    QC outputs live there and must never be re-ingested as scans).
    """
    files: list[Path] = []
    try:
        for root, dirnames, filenames in os.walk(directory):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "qc"]
            for name in filenames:
                if Path(name).suffix.lower() in SCAN_EXTENSIONS:
                    files.append(Path(root) / name)
    except OSError:
        pass
    return sorted(files)


def _ensure_jpeg(root: Path, filepath: Path) -> Path:
    """
    Cache a web-sized JPEG under <root>/qc/cache/ (reused by the UI).

    The cache name encodes the path relative to the monitored root so
    same-named files in different subdirectories don't collide.
    """
    if filepath.suffix.lower() in (".jpg", ".jpeg"):
        return filepath
    try:
        rel = filepath.relative_to(root)
    except ValueError:
        rel = Path(filepath.name)
    flat = "__".join(rel.parts[:-1] + (rel.stem,)) + ".jpg"
    jp = root / "qc" / "cache" / flat
    if jp.exists() and jp.stat().st_mtime >= filepath.stat().st_mtime:
        return jp
    return convert_tiff_to_jpeg(filepath, jp)


def _find_root(filepath: Path, roots: list[Path]) -> Path:
    """The monitored root containing *filepath* (falls back to its parent)."""
    for root in sorted(roots, key=lambda r: len(str(r)), reverse=True):
        if filepath.is_relative_to(root):
            return root
    return filepath.parent


def _collect_work(directories: list[str], rerun_of: int | None,
                  paths: list[str] | None = None) -> list[tuple[Path, Path]]:
    """(monitored_root, filepath) pairs to process."""
    if paths is not None:
        # Explicit file list (folder- or image-level run from the
        # Directories view). No seen-filter: re-running processed files
        # deliberately produces fresh scan records in the new batch.
        roots = [Path(p["path"]) for p in db.list_projects()]
        return [(_find_root(p, roots), p)
                for p in (Path(x) for x in paths) if p.exists()]

    if rerun_of is not None:
        # Re-pass: same file set as the original batch (files still on disk).
        roots = [Path(p["path"]) for p in db.list_projects()]
        file_paths = [Path(s["filepath"]) for s in db.list_scans(rerun_of)]
        return [(_find_root(p, roots), p) for p in file_paths if p.exists()]

    seen = db.already_scanned_paths()
    work: list[tuple[Path, Path]] = []
    for d in directories:
        root = Path(d)
        for f in discover_files(root):
            if str(f) not in seen:
                work.append((root, f))
    return work


def preview_work(directories: list[str] | None = None) -> dict:
    """
    Dry-run directory scan: how many images each monitored directory
    holds and how many are new (never processed). No side effects.
    """
    if directories is None:
        directories = [p["path"] for p in db.list_projects(enabled_only=True)]
    seen = db.already_scanned_paths()

    dirs = []
    for d in directories:
        files = discover_files(Path(d))
        new = sum(1 for f in files if str(f) not in seen)
        dirs.append({"path": d, "total": len(files), "new": new})
    return {
        "directories": dirs,
        "total": sum(x["total"] for x in dirs),
        "new": sum(x["new"] for x in dirs),
    }


def run_batch(trigger: str = "manual", directories: list[str] | None = None,
              rerun_of: int | None = None,
              badscan_model_id: int | None = None,
              pagetype_model_id: int | None = None,
              paths: list[str] | None = None,
              project_id: int | None = None) -> int | None:
    """
    Run one QC pass. Returns the batch id, or None if a batch is already
    running. Blocking — call from a worker thread for the web app.

    Model ids default to the active version of each role; passing
    explicit ids runs the pass with those versions instead. *paths*
    restricts the pass to an explicit file list (bypassing the
    new-files-only filter) for folder- and image-level runs.
    *project_id* attributes the pass to one project; when omitted it is
    inferred from a single-directory *directories*.
    """
    if not _batch_lock.acquire(blocking=False):
        return None

    global progress
    batch_id = None
    try:
        _cancel.clear()
        settings = db.get_settings()
        if directories is None:
            directories = [p["path"] for p in db.list_projects(enabled_only=True)]

        if badscan_model_id is None:
            active = db.get_active_model("badscan")
            badscan_model_id = active["id"] if active else None
        if pagetype_model_id is None:
            active = db.get_active_model("pagetype")
            pagetype_model_id = active["id"] if active else None
        backend = settings.get("transcription_backend", "off")

        batch_id = db.create_batch(
            trigger=trigger,
            directories=directories,
            badscan_model_id=badscan_model_id,
            pagetype_model_id=pagetype_model_id,
            transcription_backend=backend,
            rerun_of=rerun_of,
            project_id=project_id,
        )

        work = _collect_work(directories, rerun_of, paths)
        transcribe_types = {
            t.strip() for t in settings.get("transcribe_page_types", "handwriting").split(",")
            if t.strip()
        }

        progress = {"batch_id": batch_id, "total": len(work), "done": 0,
                    "current": "", "status": "running"}
        print(f"[batch {batch_id}] {trigger}: {len(work)} file(s) to process")

        # Optional: batch the YOLO inference on the remote GPU box.
        remote_preds = None
        if work and _remote_inference_enabled():
            remote_preds = _remote_infer_all(work, badscan_model_id, pagetype_model_id)

        backend_on = settings.get("transcription_backend", "off") not in ("", "off")
        pending: list[tuple[int, Path]] = []
        cancelled = False
        for root, filepath in work:
            if _cancel.is_set():
                cancelled = True
                break
            progress["current"] = filepath.name
            pre = remote_preds.get(str(filepath)) if remote_preds is not None else None
            res = _process_one(batch_id, root, filepath, settings,
                               badscan_model_id, pagetype_model_id,
                               remote_pre=pre, remote_mode=remote_preds is not None)
            if res and backend_on and res[2] in transcribe_types:
                pending.append((res[0], res[1]))
            progress["done"] += 1

        if not cancelled:
            _transcribe_many(pending, settings)
            cancelled = _cancel.is_set()

        db.finish_batch(batch_id, status="cancelled" if cancelled else "done")
        progress = {}
        print(f"[batch {batch_id}] {'cancelled' if cancelled else 'done'}")
        return batch_id

    except Exception as e:
        traceback.print_exc()
        if batch_id is not None:
            db.finish_batch(batch_id, status="error", error=str(e))
        progress = {}
        return batch_id
    finally:
        _batch_lock.release()


def classify_page_full(image, jp: Path, settings: dict,
                       pagetype_model_id: int | None = None,
                       precomputed: tuple | None = None
                       ) -> tuple[str | None, float, str]:
    """
    Page-type decision rule: the trained local model is authoritative for
    the normal (built-in) classes. The LLM endpoint is only used for
    custom classes the local model hasn't learned yet — or for everything
    while no trained local model is active at all.

    *precomputed* is (class_or_None, conf) from remote inference; when
    given, the model step is skipped and *image* is not used.
    Returns (page_type, confidence, source).
    """
    if precomputed is not None:
        page_type, page_conf = precomputed
    else:
        page_type, page_conf = inference.classify_page(image, pagetype_model_id)
    page_source = "model" if page_type is not None else ""

    known = inference.model_classes("pagetype", pagetype_model_id) or set()
    custom = set(all_page_types(settings)) - set(PAGE_TYPES)
    missing_custom = custom - known

    if page_type is None or missing_custom:
        try:
            llm_label, llm_backend = transcribe.classify_page_llm(jp, settings)
            if llm_label is not None and (page_type is None or llm_label in missing_custom):
                # Use the LLM label when there is no local prediction, or
                # when the LLM recognises a custom class the local model
                # can't produce yet.
                return llm_label, 1.0, f"llm:{llm_backend}"
        except Exception as ce:
            print(f"  LLM page classification failed for {jp.name}: {ce}")

    return page_type, page_conf, page_source


VALID_STAGES = {"badscan", "pagetype", "transcription"}


def rerun_stages(scan_ids: list[int], stages: set[str],
                 force_transcribe: bool = False) -> dict | None:
    """
    Re-run selected stages over existing scan rows, updating them in
    place with the currently active models / configured backend.

    Returns {"updated": n, "cancelled": bool}, or None if a batch or
    another stage re-run is already in progress. Blocking — run in a
    worker thread from the web app for more than a scan or two.
    """
    stages = stages & VALID_STAGES
    if not stages:
        return {"updated": 0, "cancelled": False}
    if not _batch_lock.acquire(blocking=False):
        return None

    global progress
    try:
        _cancel.clear()
        settings = db.get_settings()
        roots = [Path(p["path"]) for p in db.list_projects()]
        stage_label = "+".join(sorted(stages))

        progress = {"total": len(scan_ids), "done": 0, "current": "",
                    "status": "running", "stage_rerun": stage_label}
        print(f"[stage-rerun] {stage_label}: {len(scan_ids)} scan(s)")

        # Transcription-only rerun: collect the work, then transcribe it
        # through _transcribe_many so a server backend gets concurrent
        # requests instead of one page at a time.
        if stages == {"transcription"}:
            transcribe_types = {
                t.strip() for t in settings.get("transcribe_page_types", "handwriting").split(",")
                if t.strip()
            }
            items: list[tuple[int, Path]] = []
            touched: set[int] = set()
            for sid in scan_ids:
                scan = db.get_scan(sid)
                if scan is None:
                    continue
                fp = Path(scan["filepath"])
                if not fp.exists():
                    db.update_scan(sid, status="error", error="File no longer exists")
                    touched.add(scan["batch_id"])
                    continue
                if not (force_transcribe or scan["page_type"] in transcribe_types):
                    continue
                try:
                    jp = _ensure_jpeg(_find_root(fp, roots), fp)
                except Exception as e:
                    db.update_scan(sid, status="error", error=str(e))
                    touched.add(scan["batch_id"])
                    continue
                items.append((sid, jp))
                touched.add(scan["batch_id"])

            updated = _transcribe_many(items, settings)
            cancelled = _cancel.is_set()
            for bid in touched:
                db.refresh_batch_stats(bid)
            progress = {}
            print(f"[stage-rerun] {'cancelled' if cancelled else 'done'} "
                  f"({updated} transcribed)")
            return {"updated": updated, "cancelled": cancelled}

        touched_batches: set[int] = set()
        updated = 0
        cancelled = False
        for sid in scan_ids:
            if _cancel.is_set():
                cancelled = True
                break
            scan = db.get_scan(sid)
            if scan is None:
                continue
            progress["current"] = scan["filename"]
            if _rerun_one(scan, stages, settings, roots, force_transcribe):
                updated += 1
                touched_batches.add(scan["batch_id"])
            progress["done"] += 1

        for bid in touched_batches:
            db.refresh_batch_stats(bid)
        progress = {}
        print(f"[stage-rerun] {'cancelled' if cancelled else 'done'} ({updated} updated)")
        return {"updated": updated, "cancelled": cancelled}
    finally:
        progress = {}
        _batch_lock.release()


def _rerun_one(scan: dict, stages: set[str], settings: dict,
               roots: list[Path], force_transcribe: bool) -> bool:
    filepath = Path(scan["filepath"])
    if not filepath.exists():
        db.update_scan(scan["id"], status="error", error="File no longer exists")
        return True

    try:
        root = _find_root(filepath, roots)
        jp = _ensure_jpeg(root, filepath)
        updates: dict = {"jpeg_path": str(jp), "status": "ok", "error": ""}

        image = None
        if stages & {"badscan", "pagetype"}:
            image = cv2.imread(str(jp))
            if image is None:
                raise ValueError("Could not read image")

        if "badscan" in stages:
            bad_scan, conf = inference.classify_badscan(image)
            updates["bad_scan"] = bad_scan
            updates["bad_scan_conf"] = conf

        page_type = scan["page_type"]
        if "pagetype" in stages:
            page_type, conf, source = classify_page_full(image, jp, settings)
            updates["page_type"] = page_type
            updates["page_type_conf"] = conf
            updates["page_type_source"] = source

        if "transcription" in stages:
            transcribe_types = {
                t.strip() for t in settings.get("transcribe_page_types", "handwriting").split(",")
                if t.strip()
            }
            if force_transcribe or page_type in transcribe_types:
                text, backend, geometry = transcribe.transcribe(jp, settings)
                updates["transcription"] = text
                updates["transcription_backend"] = backend
                updates["transcription_lines"] = json.dumps(geometry) if geometry else None
                if text is not None:
                    db.save_transcription(scan["id"], backend,
                                          text, updates["transcription_lines"])

        db.update_scan(scan["id"], **updates)
        return True
    except Exception as e:
        db.update_scan(scan["id"], status="error", error=str(e))
        return True


def _remote_inference_enabled() -> bool:
    from . import remote
    if not remote.inference_enabled():
        return False
    if remote.reachable():
        return True
    print("  remote inference server unreachable — using local inference")
    return False


def _deployed_versions(badscan_model_id, pagetype_model_id) -> dict[str, str]:
    """Ensure the chosen (and task-ready) models are on the remote; return role->version."""
    from . import remote
    out = {}
    for role, mid in (("badscan", badscan_model_id), ("pagetype", pagetype_model_id)):
        rec = db.get_model(mid) if mid else db.get_active_model(role)
        if rec and inference.model_ready(role, rec["id"]) and Path(rec["path"]).exists():
            try:
                remote.ensure_model(rec["version"], Path(rec["path"]))
                out[role] = rec["version"]
            except Exception as e:
                print(f"  remote deploy of {rec['version']} failed: {e}")
    return out


def _remote_infer_all(work: list, badscan_model_id, pagetype_model_id,
                      chunk: int = 24) -> dict[str, dict]:
    """
    Batch-classify every work image on the remote box, keyed by filepath.
    Reads each image once. Roles whose model isn't trained/deployed are
    simply absent — the per-scan step then falls back (LLM for pagetype).
    """
    from . import remote

    global progress
    deployed = _deployed_versions(badscan_model_id, pagetype_model_id)
    preds: dict[str, dict] = {}
    if not deployed:
        return preds  # nothing ready remotely; loop will handle locally/LLM

    progress["status"] = "remote inference"
    for i in range(0, len(work), chunk):
        if _cancel.is_set():
            break
        batch = work[i:i + chunk]
        imgs, paths = [], []
        for root, fp in batch:
            try:
                jp = _ensure_jpeg(root, fp)
                img = cv2.imread(str(jp))
                if img is None:
                    continue
                imgs.append(img)
                paths.append(str(fp))
            except Exception:
                continue
        if not imgs:
            continue
        entry = {p: {} for p in paths}
        for role, ver in deployed.items():
            try:
                res = remote.infer_batch(ver, imgs)
                for p, r in zip(paths, res):
                    entry[p][role] = (r.get("class"), float(r.get("conf") or 0.0))
            except Exception as e:
                print(f"  remote {role} inference failed on chunk: {e}")
        preds.update(entry)
        progress["done"] = min(len(work), i + len(batch))
    progress["status"] = "running"
    progress["done"] = 0
    return preds


def _process_one(batch_id: int, root: Path, filepath: Path, settings: dict,
                 badscan_model_id: int | None = None,
                 pagetype_model_id: int | None = None,
                 remote_pre: dict | None = None,
                 remote_mode: bool = False) -> tuple[int, Path, str | None] | None:
    """
    Classify one file and record its scan row (transcription left empty —
    the batch transcribes afterwards, concurrently where the backend
    allows). Returns (scan_id, jpeg_path, page_type), or None on error.
    """
    directory = str(filepath.parent)
    try:
        jp = _ensure_jpeg(root, filepath)

        if remote_mode:
            pre = remote_pre or {}
            bad_scan, bad_scan_conf = pre.get("badscan", (None, 0.0))
            page_type, page_conf, page_source = classify_page_full(
                None, jp, settings, pagetype_model_id,
                precomputed=pre.get("pagetype", (None, 0.0)))
        else:
            image = cv2.imread(str(jp))
            if image is None:
                raise ValueError("Could not read image")
            bad_scan, bad_scan_conf = inference.classify_badscan(image, badscan_model_id)
            page_type, page_conf, page_source = classify_page_full(
                image, jp, settings, pagetype_model_id)

        scan_id = db.add_scan(
            batch_id, directory, filepath.name, str(filepath),
            jpeg_path=str(jp),
            bad_scan=bad_scan, bad_scan_conf=bad_scan_conf,
            page_type=page_type, page_type_conf=page_conf,
            page_type_source=page_source,
        )
        return scan_id, jp, page_type
    except Exception as e:
        db.add_scan(
            batch_id, directory, filepath.name, str(filepath),
            status="error", error=str(e),
        )
        return None


def _transcribe_many(items: list[tuple[int, Path]], settings: dict) -> int:
    """
    Transcribe (scan_id, jpeg_path) pairs and store each result on the scan
    and in the per-backend transcriptions table.

    Runs transcribe.max_concurrency(settings) pages at a time: a vLLM /
    SGLang / Vertex endpoint batches concurrent requests on the server, so
    parallel in-flight requests are what keeps its GPU busy. Local engines
    are declared with concurrency 1 and run exactly as before. Failures
    are per-page and never sink the pass; cancellation is honoured between
    pages. Returns how many pages got a transcription.
    """
    if not items:
        return 0
    from concurrent.futures import ThreadPoolExecutor

    global progress
    workers = min(transcribe.max_concurrency(settings), len(items))
    progress.update(total=len(items), done=0,
                    status=f"transcribing ({workers} at a time)")
    print(f"  transcribing {len(items)} page(s), {workers} at a time")
    done_lock = threading.Lock()
    n_ok = 0

    def one(item: tuple[int, Path]) -> bool:
        nonlocal n_ok
        sid, jp = item
        if _cancel.is_set():
            return False
        progress["current"] = jp.name
        ok = False
        try:
            text, backend, geometry = transcribe.transcribe(jp, settings)
            lines_json = json.dumps(geometry) if geometry else None
            db.update_scan(sid, transcription=text,
                           transcription_backend=backend,
                           transcription_lines=lines_json)
            if text is not None:
                db.save_transcription(sid, backend, text, lines_json)
                ok = True
        except Exception as te:
            print(f"  transcription failed for {jp.name}: {te}")
        with done_lock:
            progress["done"] += 1
            if ok:
                n_ok += 1
        return ok

    if workers == 1:
        for item in items:
            one(item)
    else:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="transcribe") as pool:
            list(pool.map(one, items))
    return n_ok
