"""
User Interface layer.

Holds everything the visitor touches:

    routes.py    page controllers (HTML)
    api.py       JSON endpoints used by the front-end and by other clients
    templates/   Jinja2 templates
    static/      hand-written CSS, vanilla JavaScript, fonts and icons

Controllers in this layer orchestrate only. All clinical logic lives in
`engine` and all clinical facts live in `knowledgebase`.
"""

from app.interface.api import api_bp
from app.interface.routes import pages_bp

__all__ = ["api_bp", "pages_bp"]
