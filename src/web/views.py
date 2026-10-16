"""Page routes for the new interface."""

from flask import Blueprint, render_template

from .. import db
from ..config import MODEL_ROLES, all_page_types

bp = Blueprint("views", __name__)


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


@bp.get("/training")
def training_page():
    """Live training: start a run and watch it, plus run history."""
    return render_template(
        "training.html",
        active_nav="training",
        roles=list(MODEL_ROLES),
    )


@bp.get("/settings")
def settings_page():
    """Editable settings, grouped by what they affect."""
    from ..config import DEFAULT_SETTINGS

    return render_template(
        "settings.html",
        active_nav="settings",
        settings=db.get_settings(),
        keys=sorted(DEFAULT_SETTINGS.keys()),
    )


@bp.get("/records")
def records_page():
    """The scan table: filter on any column, see it as CSV, export it."""
    return render_template("records.html", active_nav="records")


@bp.get("/data")
def data_browser():
    """The training-data browser: inspect and correct every labeled image."""
    return render_template(
        "data.html",
        active_nav="data",
        roles=list(MODEL_ROLES),
        projects=db.list_projects(),
    )
