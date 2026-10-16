"""
A tabular view of QC results: one row per scan, filterable on any column and
exportable as CSV.

The rows are the *latest* scan of every file the app has processed, so the
table keeps a record of what passed and what did not even after the file has
left the staging folder. Each row joins what the model said, what an
annotator said, what became of the image in the training data, and the
transcription, as flat columns.

Columns are defined once, in :data:`COLUMNS`. Every filter, sort, and CSV
header is checked against that list, and values only ever reach SQL as bound
parameters — column names come from the list, never from the request.

Two derived columns deserve a definition, since they answer "what passed":

  result      passed / not_passed / unscored. An annotator's answer wins
              when there is one; otherwise it is what the model said.
  model_outcome   how the model fared once reviewed: confirmed_bad,
              false_alarm (flagged, but fine), missed_bad (not flagged,
              but bad), confirmed_clean, or not_reviewed.
"""

import csv
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date

from . import db

# Distinct values offered per choice column before it is treated as too
# varied to list (the filter then falls back to what the user types).
FACET_CAP = 300
MAX_PER_PAGE = 500
CHOICE_BLANK = ""   # a choice filter value of "" means "(blank)"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class FilterError(ValueError):
    """A request named a column or filter the table does not have."""


@dataclass(frozen=True)
class Col:
    key: str          # CSV header and API name
    sql: str          # expression over the aliases in _FROM
    kind: str         # text | choice | number | date
    group: str
    desc: str
    default: bool = False   # shown (and exported) unless the user changes it
    search: bool = False    # covered by the "search all text" box
    facet_order: str = "count"   # count | value_desc


def _example(select: str, role: str | None = None) -> str:
    """Most recent active training example of this scan (optionally one role)."""
    where = "te.scan_id = s.id AND te.active = 1"
    if role:
        where += f" AND te.role = '{role}'"
    return (f"(SELECT {select} FROM training_examples te WHERE {where} "
            "ORDER BY te.id DESC LIMIT 1)")


_ANSWERED = "r.verdict IN ('true_positive','false_negative','false_positive','true_negative')"

COLUMNS: list[Col] = [
    # --- the scan -----------------------------------------------------------
    Col("scan_id", "s.id", "number", "Scan",
        "Internal id of the latest scan of this file."),
    Col("project", "p.name", "choice", "Scan",
        "The project (monitored folder) the file belongs to.", True, True),
    Col("folder", "NULLIF(CASE WHEN p.id IS NULL THEN s.directory "
        "WHEN s.directory = p.path THEN '' "
        "ELSE substr(s.directory, length(p.path) + 2) END, '')", "text", "Scan",
        "Folder of the file, relative to its project's root.", True, True),
    Col("filename", "s.filename", "text", "Scan", "File name.", True, True),
    Col("filepath", "s.filepath", "text", "Scan",
        "Full path the file was processed from.", False, True),
    Col("on_disk", "CASE WHEN p.id IS NULL THEN NULL WHEN s.present = 1 THEN 'yes' "
        "ELSE 'no' END", "choice", "Scan",
        "Whether the file was still in its folder at the last pass over that "
        "folder. Approved scans move on to their permanent home. Blank for "
        "older imports whose folder is no longer a project, since nothing "
        "tracks it.", True),
    Col("pass", "b.label", "choice", "Scan",
        "The pass (batch) that produced this result.", False, False, "value_desc"),
    Col("processed_at", "s.processed_at", "date", "Scan",
        "When the file was last processed (UTC)."),
    Col("status", "s.status", "choice", "Scan", "ok, or error if it could not be read."),
    Col("error", "NULLIF(s.error, '')", "text", "Scan",
        "Why processing failed, when it did.", False, True),
    Col("file_size_bytes", "NULLIF(s.file_size, 0)", "number", "Scan",
        "Size of the file when it was processed."),

    # --- the model ----------------------------------------------------------
    Col("model_flagged", "CASE s.bad_scan WHEN 'bad_scan' THEN 'yes' "
        "WHEN 'no_bad_scan' THEN 'no' END", "choice", "Model",
        "Did the bad-scan model flag it? Blank means it was not scored.", True),
    Col("model_confidence", "CASE WHEN s.bad_scan IS NULL THEN NULL "
        "ELSE round(s.bad_scan_conf, 4) END", "number", "Model",
        "How sure the model was of its own answer, 0 to 1.", True),
    Col("model_page_type", "s.page_type", "choice", "Model",
        "Page type the model (or LLM) guessed."),
    Col("model_page_type_confidence", "CASE WHEN s.page_type IS NULL THEN NULL "
        "ELSE round(s.page_type_conf, 4) END", "number", "Model",
        "Confidence in that page-type guess, 0 to 1."),
    Col("model_page_type_source", "NULLIF(s.page_type_source, '')", "choice", "Model",
        "Where the page-type guess came from: model or an LLM backend."),
    Col("bad_scan_model", "m1.version", "choice", "Model",
        "Bad-scan model version the pass ran with.", False, False, "value_desc"),
    Col("page_type_model", "m2.version", "choice", "Model",
        "Page-type model version the pass ran with.", False, False, "value_desc"),

    # --- the annotator ------------------------------------------------------
    Col("result", "CASE WHEN r.verdict IN ('true_positive','false_negative') THEN 'not_passed' "
        "WHEN r.verdict IN ('false_positive','true_negative') THEN 'passed' "
        "WHEN s.bad_scan = 'bad_scan' THEN 'not_passed' "
        "WHEN s.bad_scan = 'no_bad_scan' THEN 'passed' ELSE 'unscored' END",
        "choice", "Annotator",
        "passed or not_passed. The annotator's answer when there is one, "
        "otherwise the model's; unscored if neither exists.", True),
    Col("result_by", f"CASE WHEN {_ANSWERED} THEN 'annotator' "
        "WHEN s.bad_scan IS NOT NULL THEN 'model' END", "choice", "Annotator",
        "Whose answer the result column reflects."),
    Col("annotator_says", "CASE WHEN r.verdict IN ('true_positive','false_negative') "
        "THEN 'bad' WHEN r.verdict IN ('false_positive','true_negative') THEN 'fine' END",
        "choice", "Annotator",
        "The annotator's answer to 'is this a bad scan?'. Blank if not reviewed.", True),
    Col("model_outcome", "CASE r.verdict WHEN 'true_positive' THEN 'confirmed_bad' "
        "WHEN 'false_positive' THEN 'false_alarm' WHEN 'false_negative' THEN 'missed_bad' "
        "WHEN 'true_negative' THEN 'confirmed_clean' ELSE 'not_reviewed' END",
        "choice", "Annotator",
        "How the model fared: confirmed_bad, false_alarm (flagged but fine), "
        "missed_bad (not flagged but bad), confirmed_clean, or not_reviewed. "
        "The two it got wrong are the ones kept for training.", True),
    Col("verdict", "NULLIF(r.verdict, '')", "choice", "Annotator",
        "The same outcome as true/false positive/negative."),
    Col("annotator_page_type", "NULLIF(r.page_type, '')", "choice", "Annotator",
        "Page type the annotator set. Blank if they did not."),
    Col("page_type", "COALESCE(NULLIF(r.page_type, ''), s.page_type)", "choice", "Annotator",
        "Final page type: the annotator's when set, otherwise the model's.", False),
    Col("reviewer", "NULLIF(r.reviewer, '')", "choice", "Annotator",
        "Who answered.", True, True),
    Col("reviewed_at", "r.created_at", "date", "Annotator",
        "When the annotator last answered or set a page type (UTC).", True),
    Col("annotator_notes", "NULLIF(r.notes, '')", "text", "Annotator",
        "The annotator's notes on the scan.", True, True),
    Col("annotation_count", "(SELECT COUNT(*) FROM annotations a "
        "WHERE a.scan_id = s.id OR a.filepath = s.filepath)", "number", "Annotator",
        "How many annotator actions are logged for this file: answers, "
        "corrections, note edits."),

    # --- training data ------------------------------------------------------
    Col("kept_for_training", "CASE WHEN EXISTS (SELECT 1 FROM training_examples te "
        "WHERE te.scan_id = s.id AND te.active = 1) THEN 'yes' ELSE 'no' END",
        "choice", "Training",
        "Is a copy of this scan kept in the training data?", True),
    Col("training_label", _example("te.label", "badscan"), "choice", "Training",
        "Current bad-scan label in the training data (annotators can correct it)."),
    Col("page_type_training_label", _example("te.label", "pagetype"), "choice", "Training",
        "Current page-type label in the training data."),
    Col("training_label_corrected",
        "CASE " + _example("CASE WHEN te.original_label <> '' AND te.label <> te.original_label "
                           "THEN 1 ELSE 0 END") + " WHEN 1 THEN 'yes' WHEN 0 THEN 'no' END",
        "choice", "Training",
        "Was a training label changed after it was first recorded?"),
    Col("kept_image_path", _example("te.image_path"), "text", "Training",
        "Where the kept copy of the image lives.", False, True),

    # --- transcription ------------------------------------------------------
    Col("transcribed", "CASE WHEN s.transcription IS NOT NULL THEN 'yes' ELSE 'no' END",
        "choice", "Transcription", "Does it have a transcription?", True),
    Col("transcription_backend", "NULLIF(s.transcription_backend, '')", "choice",
        "Transcription", "Which backend transcribed it."),
    Col("transcription_chars", "length(s.transcription)", "number", "Transcription",
        "Length of the transcription in characters."),
    Col("transcription", "s.transcription", "text", "Transcription",
        "The transcribed text.", False, True),
]

BY_KEY = {c.key: c for c in COLUMNS}
DEFAULT_COLUMNS = [c.key for c in COLUMNS if c.default]
CHOICE_KEYS = [c.key for c in COLUMNS if c.kind == "choice"]

_FROM = """
FROM scans s
JOIN (SELECT filepath, MAX(id) AS mid FROM scans GROUP BY filepath) t ON s.id = t.mid
JOIN batches b ON b.id = s.batch_id
LEFT JOIN models m1 ON m1.id = b.badscan_model_id
LEFT JOIN models m2 ON m2.id = b.pagetype_model_id
LEFT JOIN reviews r ON r.scan_id = s.id
LEFT JOIN projects p ON p.id = (
    SELECT p2.id FROM projects p2
    WHERE s.directory = p2.path
       OR (p2.mode = 'tree'
           AND substr(s.directory, 1, length(p2.path) + 1) = p2.path || '/')
    ORDER BY length(p2.path) DESC LIMIT 1)
"""

# Every column, aliased to its key, so filters and sorts can name columns
# as plain identifiers on the outside.
_BASE = "SELECT " + ", ".join(f'{c.sql} AS "{c.key}"' for c in COLUMNS) + _FROM

# Preset views: server-owned so their column names cannot drift from COLUMNS.
PRESETS = [
    {"id": "all", "label": "All scans", "filters": {},
     "hint": "Every file the app has processed, including ones that have moved on."},
    {"id": "passed", "label": "Passed", "filters": {"result": {"in": ["passed"]}},
     "hint": "Fine according to the annotator, or the model if nobody reviewed it."},
    {"id": "not_passed", "label": "Not passed",
     "filters": {"result": {"in": ["not_passed"]}},
     "hint": "A bad scan according to the annotator, or the model if nobody reviewed it."},
    {"id": "flagged", "label": "Model flagged",
     "filters": {"model_flagged": {"in": ["yes"]}},
     "hint": "The bad-scan model flagged it, whatever an annotator later said."},
    {"id": "model_wrong", "label": "Model was wrong",
     "filters": {"model_outcome": {"in": ["false_alarm", "missed_bad"]}},
     "hint": "An annotator overruled the model. These are kept as training data."},
    {"id": "needs_review", "label": "Awaiting review",
     "filters": {"annotator_says": {"in": [CHOICE_BLANK]}, "on_disk": {"in": ["yes"]}},
     "hint": "Still in the folder, with no annotator answer yet."},
    {"id": "kept", "label": "Kept for training",
     "filters": {"kept_for_training": {"in": ["yes"]}},
     "hint": "A copy is kept in the training data."},
    {"id": "gone", "label": "Left the folder", "filters": {"on_disk": {"in": ["no"]}},
     "hint": "Seen on an earlier pass, no longer in its folder."},
]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _as_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise FilterError(f"Not a number: {v!r}") from None


def _clean_spec(col: Col, spec: dict) -> dict:
    """Validate one column's filter, returning only what it needs."""
    if col.kind == "choice":
        vals = spec.get("in")
        if not isinstance(vals, list) or not vals:
            raise FilterError(f"{col.key}: pick at least one value")
        return {"in": [str(v) for v in vals[:FACET_CAP * 2]]}
    if col.kind == "text":
        text = str(spec.get("contains") or "").strip()
        if not text:
            raise FilterError(f"{col.key}: enter some text to look for")
        return {"contains": text}
    if col.kind == "number":
        out = {k: _as_float(spec.get(k)) for k in ("min", "max")}
        out = {k: v for k, v in out.items() if v is not None}
        if not out:
            raise FilterError(f"{col.key}: enter a minimum or a maximum")
        return out
    out = {}
    for k in ("min", "max"):
        v = spec.get(k)
        if v in (None, ""):
            continue
        if not isinstance(v, str) or not _DATE_RE.match(v):
            raise FilterError(f"{col.key}: dates look like 2026-10-15")
        out[k] = v
    if not out:
        raise FilterError(f"{col.key}: enter a start or an end date")
    return out


def parse_filters(raw: str | None) -> dict:
    """Parse and validate the ``filters`` request parameter (a JSON object)."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        raise FilterError("filters must be a JSON object") from None
    if not isinstance(data, dict):
        raise FilterError("filters must be a JSON object")
    out = {}
    for key, spec in data.items():
        col = BY_KEY.get(key)
        if col is None:
            raise FilterError(f"Unknown column: {key}")
        if not isinstance(spec, dict):
            raise FilterError(f"{key}: malformed filter")
        out[key] = _clean_spec(col, spec)
    return out


def build_where(filters: dict, q: str = "") -> tuple[str, list]:
    """WHERE clause (possibly empty) and its bound parameters."""
    parts: list[str] = []
    args: list = []
    for key, spec in filters.items():
        col = f'"{BY_KEY[key].key}"'
        if "in" in spec:
            vals = [v for v in spec["in"] if v != CHOICE_BLANK]
            ors = []
            if vals:
                ors.append(f"{col} IN ({','.join('?' * len(vals))})")
                args += vals
            if CHOICE_BLANK in spec["in"]:
                ors.append(f"({col} IS NULL OR {col} = '')")
            parts.append("(" + " OR ".join(ors) + ")")
        elif "contains" in spec:
            # instr() rather than LIKE: no wildcard characters to escape.
            parts.append(f"instr(lower(COALESCE({col}, '')), lower(?)) > 0")
            args.append(spec["contains"])
        elif BY_KEY[key].kind == "date":
            if "min" in spec:
                parts.append(f"substr({col}, 1, 10) >= ?")
                args.append(spec["min"])
            if "max" in spec:
                parts.append(f"substr({col}, 1, 10) <= ?")
                args.append(spec["max"])
        else:
            if "min" in spec:
                parts.append(f"{col} >= ?")
                args.append(spec["min"])
            if "max" in spec:
                parts.append(f"{col} <= ?")
                args.append(spec["max"])
    q = (q or "").strip()
    if q:
        ors = [f"instr(lower(COALESCE(\"{c.key}\", '')), lower(?)) > 0"
               for c in COLUMNS if c.search]
        parts.append("(" + " OR ".join(ors) + ")")
        args += [q] * len(ors)
    return (" WHERE " + " AND ".join(parts)) if parts else "", args


def _order(sort: str | None, direction: str | None) -> str:
    key = sort if sort in BY_KEY else "processed_at"
    d = "ASC" if (direction or "").lower() == "asc" else "DESC"
    # Blanks last either way; scan_id keeps paging stable across ties.
    return f' ORDER BY ("{key}" IS NULL), "{key}" {d}, "scan_id" DESC'


def clean_columns(keys: list[str] | None) -> list[str]:
    """Requested columns in the order given, unknown and repeated ones dropped."""
    seen, out = set(), []
    for k in keys or []:
        if k in BY_KEY and k not in seen:
            seen.add(k)
            out.append(k)
    return out or list(DEFAULT_COLUMNS)


def _select(columns: list[str]) -> str:
    return "SELECT " + ", ".join(f'"{k}"' for k in columns) + f" FROM ({_BASE})"


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def total_rows() -> int:
    with db.get_db() as c:
        return c.execute(f"SELECT COUNT(*) FROM ({_BASE})").fetchone()[0]


def query(columns: list[str], filters: dict, q: str = "", sort: str | None = None,
          direction: str | None = None, limit: int = 50, offset: int = 0
          ) -> tuple[list[dict], int]:
    """One page of rows, and how many rows match in all."""
    where, args = build_where(filters, q)
    limit = max(1, min(int(limit), MAX_PER_PAGE))
    with db.get_db() as c:
        total = c.execute(f"SELECT COUNT(*) FROM ({_BASE}){where}", args).fetchone()[0]
        rows = c.execute(
            _select(columns) + where + _order(sort, direction) + " LIMIT ? OFFSET ?",
            [*args, limit, max(0, int(offset))]).fetchall()
    return [dict(r) for r in rows], total


def facets() -> dict[str, dict]:
    """
    Distinct values (with counts) of every choice column, in one pass over
    the table. A column with more than FACET_CAP values is truncated and
    marked so, since a list that long is no use as a picker.
    """
    counters = {k: Counter() for k in CHOICE_KEYS}
    cols = ", ".join(f'"{k}"' for k in CHOICE_KEYS)
    with db.get_db() as c:
        for row in c.execute(f"SELECT {cols} FROM ({_BASE})"):
            for k in CHOICE_KEYS:
                v = row[k]
                counters[k][CHOICE_BLANK if v is None or v == "" else str(v)] += 1
    out = {}
    for k, counter in counters.items():
        if BY_KEY[k].facet_order == "value_desc":
            items = sorted(counter.items(), key=lambda kv: kv[0], reverse=True)
        else:
            items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        out[k] = {"values": [{"value": v, "count": n} for v, n in items[:FACET_CAP]],
                  "truncated": len(items) > FACET_CAP}
    return out


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_FORMULA_LEADS = ("=", "+", "-", "@", "\t", "\r")


def _cell(v):
    """
    A value as a CSV cell. Text that a spreadsheet would run as a formula
    (it starts with = + - @) is prefixed with an apostrophe, so a note or a
    transcription cannot turn into one when the file is opened in Excel.
    """
    if v is None:
        return ""
    if isinstance(v, str) and v.startswith(_FORMULA_LEADS):
        return "'" + v
    return v


def csv_filename() -> str:
    return f"qc-records-{date.today().isoformat()}.csv"


def iter_csv(columns: list[str], filters: dict, q: str = "", sort: str | None = None,
             direction: str | None = None, limit: int | None = None):
    """
    Yield the CSV, header first, in chunks. The file starts with a byte-order
    mark so Excel reads the UTF-8 (manuscript transcriptions) correctly.
    Streams from the database, so a large export does not sit in memory.
    """
    where, args = build_where(filters, q)
    sql = _select(columns) + where + _order(sort, direction)
    if limit is not None:
        sql += " LIMIT ?"
        args = [*args, max(0, int(limit))]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    yield "﻿" + buf.getvalue()

    with db.get_db() as c:
        cur = c.execute(sql, args)
        while True:
            chunk = cur.fetchmany(500)
            if not chunk:
                break
            buf.seek(0)
            buf.truncate()
            for row in chunk:
                writer.writerow([_cell(row[k]) for k in columns])
            yield buf.getvalue()
