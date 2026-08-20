"""Page routes for the new interface."""

from flask import Blueprint, render_template

from .. import db
from ..config import MODEL_ROLES, all_page_types

bp = Blueprint("views", __name__, url_prefix="/ff")


@bp.get("/")
def home():
    """Project list — one card per monitored directory."""
    return render_template("projects.html", active_nav="projects")


@bp.get("/project/<int:project_id>")
def project_detail(project_id: int):
    """One project: its scans, their verdicts, and reprocess controls."""
    from flask import abort

    project = db.get_project(project_id)
    if project is None:
        abort(404)
    settings = db.get_settings()
    return render_template(
        "project.html",
        active_nav="projects",
        project=project,
        page_types=all_page_types(settings),
    )


@bp.get("/data")
def data_browser():
    """The training-data browser: inspect and correct every labeled image."""
    return render_template(
        "data.html",
        active_nav="data",
        roles=list(MODEL_ROLES),
        projects=db.list_projects(),
    )
