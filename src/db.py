"""
SQLite persistence for the batch QC workflow.

Everything the pipeline produces (batches, per-scan predictions,
transcriptions) and everything the reviewer decides (reviews) is stored
here so it can be exported as training data for the next model version.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH, DEFAULT_SETTINGS

_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS directories (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    path     TEXT NOT NULL UNIQUE,
    enabled  INTEGER NOT NULL DEFAULT 1,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT NOT NULL,              -- 'badscan' | 'pagetype'
    version    TEXT NOT NULL,              -- e.g. 'badscan-v003'
    path       TEXT NOT NULL,              -- weights file
    base_model TEXT NOT NULL DEFAULT '',   -- what it was fine-tuned from
    created_at TEXT NOT NULL,
    metrics    TEXT NOT NULL DEFAULT '{}', -- JSON (accuracy, dataset counts…)
    notes      TEXT NOT NULL DEFAULT '',
    is_active  INTEGER NOT NULL DEFAULT 0,
    UNIQUE(role, version)
);

CREATE TABLE IF NOT EXISTS batches (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    label             TEXT NOT NULL,       -- e.g. 'pass-0004'
    trigger           TEXT NOT NULL,       -- 'scheduled' | 'manual' | 'rerun'
    rerun_of          INTEGER,             -- batch id this is a re-pass of
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL DEFAULT 'running',  -- running|done|error
    error             TEXT NOT NULL DEFAULT '',
    badscan_model_id  INTEGER,
    pagetype_model_id INTEGER,
    transcription_backend TEXT NOT NULL DEFAULT 'off',
    directories       TEXT NOT NULL DEFAULT '[]',       -- JSON list of paths
    total             INTEGER NOT NULL DEFAULT 0,
    n_bad             INTEGER NOT NULL DEFAULT 0,
    n_clean           INTEGER NOT NULL DEFAULT 0,
    n_unknown         INTEGER NOT NULL DEFAULT 0,
    n_errors          INTEGER NOT NULL DEFAULT 0,
    n_transcribed     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scans (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id       INTEGER NOT NULL REFERENCES batches(id),
    directory      TEXT NOT NULL,
    filename       TEXT NOT NULL,
    filepath       TEXT NOT NULL,          -- absolute path at scan time
    jpeg_path      TEXT NOT NULL DEFAULT '',
    bad_scan       TEXT,                   -- 'bad_scan' | 'no_bad_scan' | NULL (unknown)
    bad_scan_conf  REAL NOT NULL DEFAULT 0,
    page_type      TEXT,                   -- one of PAGE_TYPES | NULL (unknown)
    page_type_conf REAL NOT NULL DEFAULT 0,
    page_type_source TEXT NOT NULL DEFAULT '',  -- 'model' | 'llm:<backend>' | ''
    transcription  TEXT,
    transcription_backend TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'ok',  -- ok | error
    error          TEXT NOT NULL DEFAULT '',
    processed_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_batch ON scans(batch_id);
CREATE INDEX IF NOT EXISTS idx_scans_path  ON scans(filepath);

CREATE TABLE IF NOT EXISTS reviews (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id    INTEGER NOT NULL UNIQUE REFERENCES scans(id),
    verdict    TEXT NOT NULL,   -- true_positive | false_positive | false_negative | true_negative
    page_type  TEXT,            -- reviewer-corrected/confirmed page type
    notes      TEXT NOT NULL DEFAULT '',
    reviewer   TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running', -- running|done|error
    error       TEXT NOT NULL DEFAULT '',
    epochs      INTEGER NOT NULL DEFAULT 0,
    dataset     TEXT NOT NULL DEFAULT '{}',      -- JSON label counts
    dataset_id  INTEGER,                         -- pinned dataset version used
    model_id    INTEGER                          -- resulting registered model
);

-- A dataset is an immutable, versioned snapshot of labeled examples used
-- for one training run. Snapshotting means a model's training data can be
-- browsed and reproduced later, even as new reviews come in.
CREATE TABLE IF NOT EXISTS datasets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT NOT NULL,              -- 'badscan' | 'pagetype'
    version    TEXT NOT NULL,              -- e.g. 'badscan-ds-v002'
    created_at TEXT NOT NULL,
    classes    TEXT NOT NULL DEFAULT '[]', -- JSON list of class names
    counts     TEXT NOT NULL DEFAULT '{}', -- JSON {class: n}
    total      INTEGER NOT NULL DEFAULT 0,
    notes      TEXT NOT NULL DEFAULT '',
    UNIQUE(role, version)
);

CREATE TABLE IF NOT EXISTS dataset_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    label      TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT '',   -- human | llm | synthetic
    image_path TEXT NOT NULL,              -- absolute path to the (jpeg) example
    split      TEXT NOT NULL DEFAULT 'train'  -- train | val
);
CREATE INDEX IF NOT EXISTS idx_dsitems_ds ON dataset_items(dataset_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _columns(db, table: str) -> set[str]:
    return {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, get_db() as db:
        db.executescript(SCHEMA)

        # --- column-add migrations (older DBs) ---
        try:
            db.execute("ALTER TABLE scans ADD COLUMN page_type_source TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("ALTER TABLE training_runs ADD COLUMN dataset_id INTEGER")
        except sqlite3.OperationalError:
            pass

        # --- mistake -> bad_scan rename migration ---
        scan_cols = _columns(db, "scans")
        if "mistake" in scan_cols and "bad_scan" not in scan_cols:
            db.execute("ALTER TABLE scans RENAME COLUMN mistake TO bad_scan")
            db.execute("ALTER TABLE scans RENAME COLUMN mistake_conf TO bad_scan_conf")
            db.execute("UPDATE scans SET bad_scan='bad_scan' WHERE bad_scan='mistake'")
            db.execute("UPDATE scans SET bad_scan='no_bad_scan' WHERE bad_scan='no_mistake'")
        batch_cols = _columns(db, "batches")
        if "mistake_model_id" in batch_cols and "badscan_model_id" not in batch_cols:
            db.execute("ALTER TABLE batches RENAME COLUMN mistake_model_id TO badscan_model_id")
            db.execute("ALTER TABLE batches RENAME COLUMN n_mistakes TO n_bad")
        # Drop stale 'mistake' seed model rows; registry.seed_registry()
        # recreates them under role 'badscan' from the base weights.
        db.execute("DELETE FROM models WHERE role='mistake'")
        db.execute("UPDATE datasets SET role='badscan' WHERE role='mistake'")
        db.execute("UPDATE training_runs SET role='badscan' WHERE role='mistake'")

        for key, value in DEFAULT_SETTINGS.items():
            db.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )


def _rows(cursor) -> list[dict]:
    return [dict(r) for r in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_settings() -> dict[str, str]:
    with get_db() as db:
        return {r["key"]: r["value"] for r in db.execute("SELECT key, value FROM settings")}


def get_setting(key: str, default: str = "") -> str:
    with get_db() as db:
        row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_settings(values: dict[str, str]) -> None:
    with _write_lock, get_db() as db:
        for key, value in values.items():
            db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

def list_directories(enabled_only: bool = False) -> list[dict]:
    q = "SELECT * FROM directories"
    if enabled_only:
        q += " WHERE enabled=1"
    q += " ORDER BY path"
    with get_db() as db:
        return _rows(db.execute(q))


def add_directory(path: str) -> dict:
    p = str(Path(path).expanduser().resolve())
    with _write_lock, get_db() as db:
        db.execute(
            "INSERT INTO directories (path, enabled, added_at) VALUES (?, 1, ?) "
            "ON CONFLICT(path) DO UPDATE SET enabled=1",
            (p, utcnow()),
        )
        return dict(db.execute("SELECT * FROM directories WHERE path=?", (p,)).fetchone())


def remove_directory(dir_id: int) -> None:
    with _write_lock, get_db() as db:
        db.execute("DELETE FROM directories WHERE id=?", (dir_id,))


def set_directory_enabled(dir_id: int, enabled: bool) -> None:
    with _write_lock, get_db() as db:
        db.execute("UPDATE directories SET enabled=? WHERE id=?", (1 if enabled else 0, dir_id))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def register_model(role: str, version: str, path: str, base_model: str = "",
                   metrics: dict | None = None, notes: str = "",
                   activate: bool = False) -> int:
    with _write_lock, get_db() as db:
        cur = db.execute(
            "INSERT INTO models (role, version, path, base_model, created_at, metrics, notes, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (role, version, path, base_model, utcnow(), json.dumps(metrics or {}), notes),
        )
        model_id = cur.lastrowid
        if activate:
            db.execute("UPDATE models SET is_active=0 WHERE role=?", (role,))
            db.execute("UPDATE models SET is_active=1 WHERE id=?", (model_id,))
        return model_id


def activate_model(model_id: int) -> dict | None:
    with _write_lock, get_db() as db:
        row = db.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        if row is None:
            return None
        db.execute("UPDATE models SET is_active=0 WHERE role=?", (row["role"],))
        db.execute("UPDATE models SET is_active=1 WHERE id=?", (model_id,))
        return dict(row)


def get_active_model(role: str) -> dict | None:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM models WHERE role=? AND is_active=1", (role,)
        ).fetchone()
        return dict(row) if row else None


def list_models(role: str | None = None) -> list[dict]:
    with get_db() as db:
        if role:
            cur = db.execute("SELECT * FROM models WHERE role=? ORDER BY id DESC", (role,))
        else:
            cur = db.execute("SELECT * FROM models ORDER BY role, id DESC")
        return _rows(cur)


def next_model_version(role: str) -> str:
    with get_db() as db:
        row = db.execute("SELECT COUNT(*) AS n FROM models WHERE role=?", (role,)).fetchone()
    return f"{role}-v{row['n'] + 1:03d}"


def get_model(model_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def create_batch(trigger: str, directories: list[str],
                 badscan_model_id: int | None, pagetype_model_id: int | None,
                 transcription_backend: str, rerun_of: int | None = None) -> int:
    with _write_lock, get_db() as db:
        row = db.execute("SELECT COUNT(*) AS n FROM batches").fetchone()
        label = f"pass-{row['n'] + 1:04d}"
        cur = db.execute(
            "INSERT INTO batches (label, trigger, rerun_of, started_at, status, "
            "badscan_model_id, pagetype_model_id, transcription_backend, directories) "
            "VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?)",
            (label, trigger, rerun_of, utcnow(), badscan_model_id,
             pagetype_model_id, transcription_backend, json.dumps(directories)),
        )
        return cur.lastrowid


_BATCH_STATS_SQL = (
    "SELECT COUNT(*) AS total,"
    " SUM(CASE WHEN bad_scan='bad_scan' THEN 1 ELSE 0 END) AS n_bad,"
    " SUM(CASE WHEN bad_scan='no_bad_scan' THEN 1 ELSE 0 END) AS n_clean,"
    " SUM(CASE WHEN bad_scan IS NULL AND status='ok' THEN 1 ELSE 0 END) AS n_unknown,"
    " SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS n_errors,"
    " SUM(CASE WHEN transcription IS NOT NULL THEN 1 ELSE 0 END) AS n_transcribed "
    "FROM scans WHERE batch_id=?"
)


def finish_batch(batch_id: int, status: str = "done", error: str = "") -> None:
    with _write_lock, get_db() as db:
        stats = db.execute(_BATCH_STATS_SQL, (batch_id,)).fetchone()
        db.execute(
            "UPDATE batches SET finished_at=?, status=?, error=?, total=?, n_bad=?, "
            "n_clean=?, n_unknown=?, n_errors=?, n_transcribed=? WHERE id=?",
            (utcnow(), status, error,
             stats["total"] or 0, stats["n_bad"] or 0, stats["n_clean"] or 0,
             stats["n_unknown"] or 0, stats["n_errors"] or 0, stats["n_transcribed"] or 0,
             batch_id),
        )


def delete_batch(batch_id: int) -> dict:
    """
    Delete a batch, its scans, and any reviews on those scans.

    Files whose only scan record was in this batch become "new" again
    and will be picked up by the next pass. Returns counts of what was
    removed so the caller can report it.
    """
    with _write_lock, get_db() as db:
        n_reviews = db.execute(
            "SELECT COUNT(*) AS n FROM reviews WHERE scan_id IN "
            "(SELECT id FROM scans WHERE batch_id=?)", (batch_id,)).fetchone()["n"]
        n_scans = db.execute(
            "SELECT COUNT(*) AS n FROM scans WHERE batch_id=?", (batch_id,)).fetchone()["n"]
        db.execute("DELETE FROM reviews WHERE scan_id IN "
                   "(SELECT id FROM scans WHERE batch_id=?)", (batch_id,))
        db.execute("DELETE FROM scans WHERE batch_id=?", (batch_id,))
        db.execute("DELETE FROM batches WHERE id=?", (batch_id,))
        return {"scans": n_scans, "reviews": n_reviews}


def count_batch_reviews(batch_id: int) -> int:
    with get_db() as db:
        return db.execute(
            "SELECT COUNT(*) AS n FROM reviews WHERE scan_id IN "
            "(SELECT id FROM scans WHERE batch_id=?)", (batch_id,)).fetchone()["n"]


def list_batches(limit: int = 100) -> list[dict]:
    with get_db() as db:
        return _rows(db.execute(
            "SELECT b.*, m1.version AS badscan_model_version, m2.version AS pagetype_model_version "
            "FROM batches b "
            "LEFT JOIN models m1 ON m1.id = b.badscan_model_id "
            "LEFT JOIN models m2 ON m2.id = b.pagetype_model_id "
            "ORDER BY b.id DESC LIMIT ?", (limit,)))


def get_batch(batch_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute(
            "SELECT b.*, m1.version AS badscan_model_version, m2.version AS pagetype_model_version "
            "FROM batches b "
            "LEFT JOIN models m1 ON m1.id = b.badscan_model_id "
            "LEFT JOIN models m2 ON m2.id = b.pagetype_model_id "
            "WHERE b.id=?", (batch_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------

def add_scan(batch_id: int, directory: str, filename: str, filepath: str,
             jpeg_path: str = "", bad_scan: str | None = None, bad_scan_conf: float = 0,
             page_type: str | None = None, page_type_conf: float = 0,
             page_type_source: str = "",
             transcription: str | None = None, transcription_backend: str = "",
             status: str = "ok", error: str = "") -> int:
    with _write_lock, get_db() as db:
        cur = db.execute(
            "INSERT INTO scans (batch_id, directory, filename, filepath, jpeg_path, "
            "bad_scan, bad_scan_conf, page_type, page_type_conf, page_type_source, "
            "transcription, transcription_backend, status, error, processed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (batch_id, directory, filename, filepath, jpeg_path, bad_scan, bad_scan_conf,
             page_type, page_type_conf, page_type_source, transcription,
             transcription_backend, status, error, utcnow()),
        )
        return cur.lastrowid


def list_scans(batch_id: int) -> list[dict]:
    with get_db() as db:
        return _rows(db.execute(
            "SELECT s.*, r.verdict AS review_verdict, r.page_type AS review_page_type, "
            " r.notes AS review_notes, r.created_at AS reviewed_at "
            "FROM scans s LEFT JOIN reviews r ON r.scan_id = s.id "
            "WHERE s.batch_id=? ORDER BY s.filename", (batch_id,)))


def get_scan(scan_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute(
            "SELECT s.*, r.verdict AS review_verdict, r.page_type AS review_page_type, "
            " r.notes AS review_notes, r.created_at AS reviewed_at "
            "FROM scans s LEFT JOIN reviews r ON r.scan_id = s.id "
            "WHERE s.id=?", (scan_id,)).fetchone()
        return dict(row) if row else None


_SCAN_UPDATE_FIELDS = {
    "jpeg_path", "bad_scan", "bad_scan_conf", "page_type", "page_type_conf",
    "page_type_source", "transcription", "transcription_backend",
    "status", "error",
}


def update_scan(scan_id: int, **fields) -> None:
    """Update selected columns of a scan row (used by per-stage reruns)."""
    cols = {k: v for k, v in fields.items() if k in _SCAN_UPDATE_FIELDS}
    if not cols:
        return
    cols["processed_at"] = utcnow()
    assignments = ", ".join(f"{k}=?" for k in cols)
    with _write_lock, get_db() as db:
        db.execute(f"UPDATE scans SET {assignments} WHERE id=?",
                   (*cols.values(), scan_id))


def refresh_batch_stats(batch_id: int) -> None:
    """Recompute a batch's aggregate counts after in-place scan updates."""
    with _write_lock, get_db() as db:
        stats = db.execute(_BATCH_STATS_SQL, (batch_id,)).fetchone()
        db.execute(
            "UPDATE batches SET total=?, n_bad=?, n_clean=?, n_unknown=?, "
            "n_errors=?, n_transcribed=? WHERE id=?",
            (stats["total"] or 0, stats["n_bad"] or 0, stats["n_clean"] or 0,
             stats["n_unknown"] or 0, stats["n_errors"] or 0,
             stats["n_transcribed"] or 0, batch_id),
        )


def already_scanned_paths() -> set[str]:
    """Absolute file paths that have been processed in any previous batch."""
    with get_db() as db:
        return {r["filepath"] for r in db.execute("SELECT DISTINCT filepath FROM scans")}


def scans_for_path(filepath: str) -> list[dict]:
    """
    Every scan version for a file across all passes, newest first —
    powers the per-image 'pass versions' view.
    """
    with get_db() as db:
        return _rows(db.execute(
            "SELECT s.*, b.label AS batch_label, b.started_at AS batch_started, "
            " m1.version AS badscan_model_version, m2.version AS pagetype_model_version, "
            " r.verdict AS review_verdict, r.page_type AS review_page_type "
            "FROM scans s JOIN batches b ON b.id = s.batch_id "
            "LEFT JOIN models m1 ON m1.id = b.badscan_model_id "
            "LEFT JOIN models m2 ON m2.id = b.pagetype_model_id "
            "LEFT JOIN reviews r ON r.scan_id = s.id "
            "WHERE s.filepath=? ORDER BY s.id DESC", (filepath,)))


def list_bad_scans() -> list[dict]:
    """
    Latest scan per file that is currently a bad scan — either the model
    flagged it (bad_scan) or a reviewer marked a missed one (false
    negative). This is the persistent bad-scan set for review + training.
    """
    with get_db() as db:
        return _rows(db.execute(
            "SELECT s.*, b.label AS batch_label, "
            " r.verdict AS review_verdict, r.page_type AS review_page_type, "
            " r.notes AS review_notes "
            "FROM scans s "
            "JOIN (SELECT filepath, MAX(id) AS mid FROM scans GROUP BY filepath) t "
            "  ON s.id = t.mid "
            "JOIN batches b ON b.id = s.batch_id "
            "LEFT JOIN reviews r ON r.scan_id = s.id "
            "WHERE s.bad_scan='bad_scan' OR r.verdict IN ('true_positive','false_negative') "
            "ORDER BY s.filepath"))


def latest_scans_by_path() -> dict[str, dict]:
    """Most recent scan record per file path (for the Directories view)."""
    with get_db() as db:
        rows = _rows(db.execute(
            "SELECT s.id, s.batch_id, s.filepath, s.bad_scan, s.page_type, s.status, "
            " s.transcription IS NOT NULL AS transcribed "
            "FROM scans s "
            "JOIN (SELECT filepath, MAX(id) AS mid FROM scans GROUP BY filepath) t "
            "ON s.id = t.mid"))
    return {r["filepath"]: r for r in rows}


# ---------------------------------------------------------------------------
# Reviews (the human-feedback signal for retraining)
# ---------------------------------------------------------------------------

def save_review(scan_id: int, verdict: str, page_type: str | None = None,
                notes: str = "", reviewer: str = "") -> None:
    with _write_lock, get_db() as db:
        db.execute(
            "INSERT INTO reviews (scan_id, verdict, page_type, notes, reviewer, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(scan_id) DO UPDATE SET verdict=excluded.verdict, "
            "page_type=excluded.page_type, notes=excluded.notes, "
            "reviewer=excluded.reviewer, created_at=excluded.created_at",
            (scan_id, verdict, page_type, notes, reviewer, utcnow()),
        )


def labeled_badscan_examples() -> list[dict]:
    """
    Human-verified binary labels derived from reviews — ALL reviewed
    scans are included, which is the whole point: every flag, missed
    flag, and correction is training signal.

    The latest review per file path wins, so re-passes after retraining
    refresh labels rather than duplicating them.
      bad_scan:    true_positive (model right) + false_negative (model missed)
      no_bad_scan: false_positive (model wrong) + true_negative (confirmed clean)
    """
    with get_db() as db:
        rows = _rows(db.execute(
            "SELECT s.filepath, s.jpeg_path, r.verdict, r.created_at "
            "FROM reviews r JOIN scans s ON s.id = r.scan_id "
            "ORDER BY r.created_at"))
    latest: dict[str, dict] = {}
    for row in rows:
        label = "bad_scan" if row["verdict"] in ("true_positive", "false_negative") else "no_bad_scan"
        latest[row["filepath"]] = {
            "filepath": row["filepath"],
            "jpeg_path": row["jpeg_path"],
            "label": label,
            "source": "human",
        }
    return list(latest.values())


def labeled_pagetype_examples(include_llm: bool = True) -> list[dict]:
    """
    Page-type training labels.

    Human-confirmed labels (latest per file path) are authoritative.
    With *include_llm*, LLM-classified scans that were never reviewed are
    added as weak labels — this bootstraps the local page-type model
    before much manual review has happened. A human label for a file
    always overrides its LLM label.
    """
    with get_db() as db:
        rows = _rows(db.execute(
            "SELECT s.filepath, s.jpeg_path, r.page_type, r.created_at "
            "FROM reviews r JOIN scans s ON s.id = r.scan_id "
            "WHERE r.page_type IS NOT NULL AND r.page_type != '' "
            "ORDER BY r.created_at"))
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row["filepath"]] = {
            "filepath": row["filepath"],
            "jpeg_path": row["jpeg_path"],
            "label": row["page_type"],
            "source": "human",
        }

    if include_llm:
        with get_db() as db:
            rows = _rows(db.execute(
                "SELECT s.filepath, s.jpeg_path, s.page_type, s.processed_at "
                "FROM scans s LEFT JOIN reviews r ON r.scan_id = s.id "
                "WHERE r.id IS NULL AND s.page_type IS NOT NULL "
                "AND s.page_type_source LIKE 'llm:%' "
                "ORDER BY s.processed_at"))
        for row in rows:
            if row["filepath"] not in latest:
                latest[row["filepath"]] = {
                    "filepath": row["filepath"],
                    "jpeg_path": row["jpeg_path"],
                    "label": row["page_type"],
                    "source": "llm",
                }
    return list(latest.values())


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------

def create_training_run(role: str, epochs: int, dataset: dict,
                        dataset_id: int | None = None) -> int:
    with _write_lock, get_db() as db:
        cur = db.execute(
            "INSERT INTO training_runs (role, started_at, status, epochs, dataset, dataset_id) "
            "VALUES (?, ?, 'running', ?, ?, ?)",
            (role, utcnow(), epochs, json.dumps(dataset), dataset_id),
        )
        return cur.lastrowid


def finish_training_run(run_id: int, status: str, error: str = "",
                        model_id: int | None = None) -> None:
    with _write_lock, get_db() as db:
        db.execute(
            "UPDATE training_runs SET finished_at=?, status=?, error=?, model_id=? WHERE id=?",
            (utcnow(), status, error, model_id, run_id),
        )


def list_training_runs(limit: int = 50) -> list[dict]:
    with get_db() as db:
        return _rows(db.execute(
            "SELECT t.*, m.version AS model_version, d.version AS dataset_version "
            "FROM training_runs t "
            "LEFT JOIN models m ON m.id = t.model_id "
            "LEFT JOIN datasets d ON d.id = t.dataset_id "
            "ORDER BY t.id DESC LIMIT ?", (limit,)))


# ---------------------------------------------------------------------------
# Datasets (immutable, versioned training snapshots)
# ---------------------------------------------------------------------------

def next_dataset_version(role: str) -> str:
    with get_db() as db:
        row = db.execute("SELECT COUNT(*) AS n FROM datasets WHERE role=?", (role,)).fetchone()
    return f"{role}-ds-v{row['n'] + 1:03d}"


def create_dataset(role: str, version: str, classes: list[str], notes: str,
                   items: list[dict]) -> int:
    """
    Persist a dataset snapshot and its items.

    *items* are dicts with label, source, image_path, split. Counts and
    total are derived and stored on the dataset row.
    """
    from collections import Counter
    counts = Counter(i["label"] for i in items)
    with _write_lock, get_db() as db:
        cur = db.execute(
            "INSERT INTO datasets (role, version, created_at, classes, counts, total, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (role, version, utcnow(), json.dumps(classes),
             json.dumps(dict(counts)), len(items), notes),
        )
        ds_id = cur.lastrowid
        db.executemany(
            "INSERT INTO dataset_items (dataset_id, label, source, image_path, split) "
            "VALUES (?, ?, ?, ?, ?)",
            [(ds_id, i["label"], i.get("source", ""), i["image_path"], i.get("split", "train"))
             for i in items],
        )
        return ds_id


def list_datasets(role: str | None = None) -> list[dict]:
    with get_db() as db:
        if role:
            cur = db.execute("SELECT * FROM datasets WHERE role=? ORDER BY id DESC", (role,))
        else:
            cur = db.execute("SELECT * FROM datasets ORDER BY role, id DESC")
        return _rows(cur)


def get_dataset(dataset_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return dict(row) if row else None


def dataset_items(dataset_id: int) -> list[dict]:
    with get_db() as db:
        return _rows(db.execute(
            "SELECT * FROM dataset_items WHERE dataset_id=? ORDER BY label, id",
            (dataset_id,)))


def delete_dataset(dataset_id: int) -> None:
    with _write_lock, get_db() as db:
        db.execute("DELETE FROM dataset_items WHERE dataset_id=?", (dataset_id,))
        db.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
