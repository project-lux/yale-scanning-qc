"""
Blueprints for the project-centric ("freeflow") interface.

The original UI is a single 3,000-line module with one inline HTML string.
New surfaces live here instead: one blueprint per area, real Jinja
templates under ``src/templates``, and per-page JS under ``src/static``.
"""

from .api_examples import bp as examples_bp
from .api_projects import bp as projects_bp
from .views import bp as views_bp

BLUEPRINTS = (projects_bp, examples_bp, views_bp)


def register(app) -> None:
    for bp in BLUEPRINTS:
        app.register_blueprint(bp)
