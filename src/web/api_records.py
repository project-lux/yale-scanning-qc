"""
Records API: the scan table behind the Records page, and its CSV export.

The page and the export take the same parameters, so the file downloaded is
exactly the rows and columns on screen:

  filters   JSON object, column -> {"in": [...]} | {"contains": "..."} |
            {"min": n, "max": n}
  q         search across every free-text column
  sort, dir column key and asc/desc
  columns   comma-separated column keys, in the order wanted
"""

from flask import Blueprint, Response, jsonify, request, stream_with_context

from .. import records

bp = Blueprint("api_records", __name__, url_prefix="/api/records")


def _params() -> dict:
    """Validated filters, search, sort and columns from the query string."""
    raw_cols = (request.args.get("columns") or "").strip()
    return dict(
        columns=records.clean_columns([c for c in raw_cols.split(",") if c]),
        filters=records.parse_filters(request.args.get("filters")),
        q=(request.args.get("q") or "").strip(),
        sort=request.args.get("sort"),
        direction=request.args.get("dir"),
    )


@bp.errorhandler(records.FilterError)
def _bad_filter(e):
    return jsonify({"error": str(e)}), 400


@bp.get("/schema")
def schema():
    """Columns, picker values for each choice column, and the preset views."""
    return jsonify({
        "columns": [{
            "key": c.key, "kind": c.kind, "group": c.group, "desc": c.desc,
            "default": c.default,
        } for c in records.COLUMNS],
        "facets": records.facets(),
        "presets": records.PRESETS,
        "total": records.total_rows(),
    })


@bp.get("")
def rows():
    """One page of rows."""
    p = _params()
    try:
        per_page = min(records.MAX_PER_PAGE, max(1, int(request.args.get("per_page") or 50)))
        page = max(1, int(request.args.get("page") or 1))
    except ValueError:
        return jsonify({"error": "page and per_page must be integers"}), 400
    data, total = records.query(
        p["columns"], p["filters"], p["q"], p["sort"], p["direction"],
        limit=per_page, offset=(page - 1) * per_page)
    return jsonify({"rows": data, "total": total, "page": page,
                    "per_page": per_page, "columns": p["columns"]})


@bp.get("/export.csv")
def export_csv():
    """
    The filtered table as a CSV download. ``limit`` returns just the first
    rows, which the page uses for its raw-CSV preview.
    """
    p = _params()
    limit = None
    if request.args.get("limit"):
        try:
            limit = max(0, int(request.args["limit"]))
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400
    gen = records.iter_csv(p["columns"], p["filters"], p["q"], p["sort"],
                           p["direction"], limit=limit)
    headers = {"Cache-Control": "no-store"}
    if limit is None:
        headers["Content-Disposition"] = f'attachment; filename="{records.csv_filename()}"'
    return Response(stream_with_context(gen), mimetype="text/csv", headers=headers)
