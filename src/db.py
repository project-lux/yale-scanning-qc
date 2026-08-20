"""
SQLite persistence for the batch QC workflow.

Everything the pipeline produces (batches, per-scan predictions,
transcriptions) and everything the reviewer decides (reviews) is stored
here so it can be exported as training data for the next model version.

Two tables carry the project-centric ("freeflow") model:

  projects          — one per monitored directory. Each directory is its own
                      project; models and datasets stay global and shared.
  training_examples — every labeled image as a first-class, editable row.
                      Previously a label was implied by which folder a file
                      sat in (data/curated/<role>/<label>/), so it could not
                      be inspected or corrected in the app. Now the row owns
                      the label and the file is just storage.
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

CREATE TABLE IF NOT EXISTS projects (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    path       TEXT NOT NULL UNIQUE,
    enabled    INTEGER NOT NULL DEFAULT 1,
    notes      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

-- One row per labeled image. `label` is the live, correctable class;
-- `original_label` keeps what it was imported as, so a correction is
-- visible rather than silently overwriting provenance. `active=0` is a
-- soft delete: the row is excluded from new datasets but stays
-- inspectable and restorable.
CREATE TABLE IF NOT EXISTS training_examples (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    role           TEXT NOT NULL,              -- 'badscan' | 'pagetype'
    label          TEXT NOT NULL,
    original_label TEXT NOT NULL DEFAULT '',
    source         TEXT NOT NULL,              -- human | curated | synthetic | llm
    image_path     TEXT NOT NULL,
    project_id     INTEGER REFERENCES projects(id),
    scan_id        INTEGER REFERENCES scans(id),
    origin_ref     TEXT NOT NULL DEFAULT '',   -- verdict json, defect kind, …
    notes          TEXT NOT NULL DEFAULT '',
    active         INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    UNIQUE(role, image_path)
);
CREATE INDEX IF NOT EXISTS idx_texamples_role   ON training_examples(role, active);
CREATE INDEX IF NOT EXISTS idx_texamples_label  ON training_examples(role, label);
CREATE INDEX IF NOT EXISTS idx_texamples_proj   ON training_examples(project_id);

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
        # Batches become project-scoped; existing rows are backfilled below.
        try:
            db.execute("ALTER TABLE batches ADD COLUMN project_id INTEGER REFERENCES projects(id)")
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

        # --- directories -> projects ---
        # The legacy `directories` table is left in place (harmless, unused)
        # so an older checkout can still read this DB.
        for d in db.execute("SELECT path, enabled, added_at FROM directories").fetchall():
            db.execute(
                "INSERT OR IGNORE INTO projects (name, path, enabled, created_at) "
                "VALUES (?, ?, ?, ?)",
                (Path(d["path"]).name or d["path"], d["path"], d["enabled"],
                 d["added_at"] or utcnow()),
            )

        # Attach existing batches to a project when their directory list names
        # exactly one known project; ambiguous multi-directory passes stay NULL
        # rather than being arbitrarily assigned.
        proj_by_path = {r["path"]: r["id"]
                        for r in db.execute("SELECT id, path FROM projects")}
        for b in db.execute(
            "SELECT id, directories FROM batches WHERE project_id IS NULL"
        ).fetchall():
            try:
                dirs = json.loads(b["directories"] or "[]")
            except Exception:
                continue
            ids = {proj_by_path[d] for d in dirs if d in proj_by_path}
            if len(ids) == 1:
                db.execute("UPDATE batches SET project_id=? WHERE id=?",
                           (ids.pop(), b["id"]))

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


# ---------------------------------------------------------------------------
# Projects (one per monitored directory)
# ---------------------------------------------------------------------------

def list_projects(enabled_only: bool = False) -> list[dict]:
    q = "SELECT * FROM projects"
    if enabled_only:
        q += " WHERE enabled=1"
    q += " ORDER BY name COLLATE NOCASE"
    with get_db() as db:
        return _rows(db.execute(q))


def get_project(project_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        return dict(row) if row else None


def get_project_by_path(path: str) -> dict | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM projects WHERE path=?", (str(path),)).fetchone()
        return dict(row) if row else None


def add_project(path: str, name: str = "") -> dict:
    path = str(path)
    with _write_lock, get_db() as db:
        db.execute(
            "INSERT OR IGNORE INTO projects (name, path, enabled, created_at) "
            "VALUES (?, ?, 1, ?)",
            (name or Path(path).name or path, path, utcnow()),
        )
    return get_project_by_path(path)


def update_project(project_id: int, **fields) -> None:
    allowed = {"name", "notes", "enabled"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    cols = ", ".join(f"{k}=?" for k in sets)
    with _write_lock, get_db() as db:
        db.execute(f"UPDATE projects SET {cols} WHERE id=?",
                   (*sets.values(), project_id))


def remove_project(project_id: int) -> None:
    """Forget a project. Scans/batches keep their rows so history survives."""
    with _write_lock, get_db() as db:
        db.execute("UPDATE training_examples SET project_id=NULL WHERE project_id=?",
                   (project_id,))
        db.execute("DELETE FROM projects WHERE id=?", (project_id,))


def project_stats(project_id: int) -> dict:
    """Counts the project list needs, in one pass per table."""
    p = get_project(project_id)
    if p is None:
        return {}
    with get_db() as db:
        scans = db.execute(
            "SELECT COUNT(*) n, "
            "SUM(bad_scan='bad_scan') n_bad, "
            "SUM(bad_scan IS NULL) n_unknown "
            "FROM scans WHERE directory=?", (p["path"],)
        ).fetchone()
        reviewed = db.execute(
            "SELECT COUNT(*) n FROM reviews r JOIN scans s ON s.id=r.scan_id "
            "WHERE s.directory=?", (p["path"],)
        ).fetchone()
        examples = db.execute(
            "SELECT COUNT(*) n FROM training_examples "
            "WHERE project_id=? AND active=1", (project_id,)
        ).fetchone()
        last = db.execute(
            "SELECT label, started_at, status FROM batches WHERE project_id=? "
            "ORDER BY id DESC LIMIT 1", (project_id,)
        ).fetchone()
    return {
        "n_scans": scans["n"] or 0,
        "n_bad": scans["n_bad"] or 0,
        "n_unknown": scans["n_unknown"] or 0,
        "n_reviewed": reviewed["n"] or 0,
        "n_examples": examples["n"] or 0,
        "last_batch": dict(last) if last else None,
    }


# ---------------------------------------------------------------------------
# Training examples — editable, first-class labeled images
# ---------------------------------------------------------------------------

def add_training_example(role: str, label: str, source: str, image_path: str,
                         project_id: int | None = None, scan_id: int | None = None,
                         origin_ref: str = "", notes: str = "") -> int | None:
    """
    Insert one labeled image. Returns its id, or None if this (role,
    image_path) is already present — re-imports never clobber a label the
    reviewer has since corrected.
    """
    now = utcnow()
    with _write_lock, get_db() as db:
        cur = db.execute(
            "INSERT OR IGNORE INTO training_examples "
            "(role, label, original_label, source, image_path, project_id, "
            " scan_id, origin_ref, notes, active, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,1,?,?)",
            (role, label, label, source, str(image_path), project_id, scan_id,
             origin_ref, notes, now, now),
        )
        return cur.lastrowid if cur.rowcount else None


def list_training_examples(role: str | None = None, label: str | None = None,
                           source: str | None = None, project_id: int | None = None,
                           active: bool | None = True, corrected_only: bool = False,
                           missing_only: bool = False,
                           limit: int = 500, offset: int = 0) -> list[dict]:
    where, args = [], []
    if role:
        where.append("te.role=?"); args.append(role)
    if label:
        where.append("te.label=?"); args.append(label)
    if source:
        where.append("te.source=?"); args.append(source)
    if project_id is not None:
        where.append("te.project_id=?"); args.append(project_id)
    if active is not None:
        where.append("te.active=?"); args.append(1 if active else 0)
    if corrected_only:
        where.append("te.original_label<>'' AND te.label<>te.original_label")
    q = ("SELECT te.*, p.name AS project_name FROM training_examples te "
         "LEFT JOIN projects p ON p.id=te.project_id")
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY te.id DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    with get_db() as db:
        rows = _rows(db.execute(q, args))
    if missing_only:
        rows = [r for r in rows if not Path(r["image_path"]).exists()]
    return rows


def count_training_examples(role: str | None = None, active: bool | None = True) -> int:
    where, args = [], []
    if role:
        where.append("role=?"); args.append(role)
    if active is not None:
        where.append("active=?"); args.append(1 if active else 0)
    q = "SELECT COUNT(*) n FROM training_examples"
    if where:
        q += " WHERE " + " AND ".join(where)
    with get_db() as db:
        return db.execute(q, args).fetchone()["n"]


def get_training_example(example_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute(
            "SELECT te.*, p.name AS project_name FROM training_examples te "
            "LEFT JOIN projects p ON p.id=te.project_id WHERE te.id=?",
            (example_id,)).fetchone()
        return dict(row) if row else None


def update_training_example(example_id: int, **fields) -> None:
    """Correct a label, edit notes, or toggle active (soft delete/restore)."""
    allowed = {"label", "notes", "active", "source"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    if "active" in sets:
        sets["active"] = 1 if sets["active"] else 0
    sets["updated_at"] = utcnow()
    cols = ", ".join(f"{k}=?" for k in sets)
    with _write_lock, get_db() as db:
        db.execute(f"UPDATE training_examples SET {cols} WHERE id=?",
                   (*sets.values(), example_id))


def bulk_update_training_examples(ids: list[int], **fields) -> int:
    allowed = {"label", "active", "source"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets or not ids:
        return 0
    if "active" in sets:
        sets["active"] = 1 if sets["active"] else 0
    sets["updated_at"] = utcnow()
    cols = ", ".join(f"{k}=?" for k in sets)
    marks = ",".join("?" * len(ids))
    with _write_lock, get_db() as db:
        cur = db.execute(
            f"UPDATE training_examples SET {cols} WHERE id IN ({marks})",
            (*sets.values(), *ids))
        return cur.rowcount


def delete_training_examples(ids: list[int], remove_files: bool = False) -> dict:
    """
    Hard-delete rows. Only unlinks the image when *remove_files* is set, and
    only for sources the app itself produced (synthetic/curated copies) —
    never a reviewer's original scan on the share.
    """
    if not ids:
        return {"deleted": 0, "files_removed": 0}
    marks = ",".join("?" * len(ids))
    removed = 0
    with get_db() as db:
        rows = _rows(db.execute(
            f"SELECT id, image_path, source FROM training_examples WHERE id IN ({marks})",
            ids))
    if remove_files:
        for r in rows:
            if r["source"] not in ("synthetic", "curated"):
                continue
            try:
                Path(r["image_path"]).unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
    with _write_lock, get_db() as db:
        cur = db.execute(f"DELETE FROM training_examples WHERE id IN ({marks})", ids)
    return {"deleted": cur.rowcount, "files_removed": removed}


def training_example_stats(role: str) -> dict:
    """Label/source breakdown for the data browser header."""
    with get_db() as db:
        by_label = {r["label"]: r["n"] for r in db.execute(
            "SELECT label, COUNT(*) n FROM training_examples "
            "WHERE role=? AND active=1 GROUP BY label", (role,))}
        by_source = {r["source"]: r["n"] for r in db.execute(
            "SELECT source, COUNT(*) n FROM training_examples "
            "WHERE role=? AND active=1 GROUP BY source", (role,))}
        agg = db.execute(
            "SELECT COUNT(*) total, "
            "SUM(active=0) inactive, "
            "SUM(original_label<>'' AND label<>original_label) corrected "
            "FROM training_examples WHERE role=?", (role,)).fetchone()
    return {
        "by_label": by_label,
        "by_source": by_source,
        "total_active": sum(by_label.values()),
        "total": agg["total"] or 0,
        "inactive": agg["inactive"] or 0,
        "corrected": agg["corrected"] or 0,
    }
