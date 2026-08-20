"""Page routes for the new interface."""

from flask import Blueprint, render_template

from .. import db
from ..config import MODEL_ROLES

bp = Blueprint("views", __name__, url_prefix="/ff")


@bp.get("/")
def home():
    """Project list — one card per monitored directory."""
    return render_template("projects.html", active_nav="projects")


@bp.get("/data")
def data_browser():
    """The training-data browser: inspect and correct every labeled image."""
    return render_template(
        "data.html",
        active_nav="data",
        roles=list(MODEL_ROLES),
        projects=db.list_projects(),
    )
