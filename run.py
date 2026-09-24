"""
Diabetes Risk Prediction Web App - application entry point.

Wires the three layers together and starts Flask:

    interface/      user interface: routes, templates, static assets
    engine/         inference: validation, model scoring, narrative generation
    knowledgebase/  WHO / IDF clinical facts and prompt scaffolding

Run:
    python run.py                       (development server)
    flask --app run run                 (equivalent)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import Flask, render_template, request

from app.config.settings import PROJECT_ROOT, get_config
from app.engine.chat import get_chat_engine
from app.engine.predictor import get_predictor
from app.interface.api import api_bp
from app.interface.routes import pages_bp
from app.knowledgebase.clinical_reference import get_reference

APP_NAME = "Diabetes Risk"

#: Navigation, shared by the header and the footer so they can never disagree.
NAV_ITEMS = [
    {"endpoint": "pages.index", "label": "Home"},
    {"endpoint": "pages.assess", "label": "Clinical assessment"},
    {"endpoint": "pages.guidelines", "label": "Clinical reference"},
    {"endpoint": "pages.model_card", "label": "Model card"},
    {"endpoint": "pages.about", "label": "About"},
]


def create_app() -> Flask:
    """Application factory."""
    config = get_config()

    logging.basicConfig(
        level=logging.DEBUG if config.is_debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # The app promises that submitted measurements are never written anywhere.
    # At DEBUG level the HTTP and LLM client libraries log full request bodies,
    # which for the narrative call means the patient's eight values and their
    # clinical interpretation land in the log file. Pin those loggers to WARNING
    # so switching on debug mode cannot quietly break that promise.
    for noisy in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    # Templates and static assets live inside the interface layer rather than
    # at the project root, so all user-facing code sits together.
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "app" / "interface" / "templates"),
        static_folder=str(PROJECT_ROOT / "app" / "interface" / "static"),
        static_url_path="/static",
    )

    app.config.update(
        SECRET_KEY=config.secret_key,
        MAX_CONTENT_LENGTH=config.max_content_length,
        JSON_SORT_KEYS=False,
        TEMPLATES_AUTO_RELOAD=config.is_debug,
        # Cookies are only used for flashed messages, but harden them anyway.
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=not config.is_debug,
    )

    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)

    _register_asset_versioning(app)
    _register_context(app, config)
    _register_security_headers(app, config)
    _register_error_handlers(app)

    _report_startup_state(logger, config)
    return app


def _register_asset_versioning(app: Flask) -> None:
    """
    Append a content version to every `url_for('static', ...)` URL.

    Static assets are served with a long cache lifetime, which would otherwise
    leave a returning visitor on a stale stylesheet after a deploy. Keying the
    URL on the file's modification time means an edited file gets a new URL and
    is refetched, while an unchanged one stays cached.
    """
    versions: dict[str, str] = {}

    @app.url_defaults
    def add_version(endpoint: str, values: dict) -> None:
        if endpoint != "static" or "filename" not in values:
            return
        filename = values["filename"]
        if filename not in versions or app.config["TEMPLATES_AUTO_RELOAD"]:
            path = PROJECT_ROOT / "app" / "interface" / "static" / filename
            try:
                versions[filename] = str(int(path.stat().st_mtime))
            except OSError:
                versions[filename] = "0"
        values["v"] = versions[filename]


def _register_context(app: Flask, config) -> None:
    """Values every template can rely on being present."""
    reference = get_reference()

    @app.context_processor
    def inject_globals():
        predictor = get_predictor()
        return {
            "app_name": APP_NAME,
            "nav_items": NAV_ITEMS,
            "current_year": datetime.now(timezone.utc).year,
            "disclaimer": reference.disclaimer,
            "model_version": predictor.metadata.get("model_version", "0.0.0"),
            "kb_reviewed": reference.raw.get("last_reviewed", ""),
            "llm_provider": config.llm_provider if config.llm_configured else None,
            # The launcher is hidden entirely when no chat-capable provider is
            # set up: an assistant that cannot answer is worse than none.
            "chat_available": get_chat_engine().is_available,
        }


def _register_security_headers(app: Flask, config) -> None:
    """Apply baseline security headers to every response."""

    @app.after_request
    def set_headers(response):
        for header, value in config.security_headers().items():
            response.headers.setdefault(header, value)

        # Static assets are safe to cache and are not health data; overriding
        # the no-store default keeps fonts and CSS from refetching every load.
        if request.path.startswith("/static/"):
            response.headers["Cache-Control"] = (
                "public, max-age=3600" if config.is_debug else "public, max-age=604800"
            )
        return response


def _register_error_handlers(app: Flask) -> None:
    """HTML error pages for browsers, JSON errors for API clients."""

    def wants_json() -> bool:
        return request.path.startswith("/api/") or request.is_json

    @app.errorhandler(404)
    def not_found(error):
        if wants_json():
            return {"error": "not_found", "message": "No such endpoint."}, 404
        return render_template("errors/404.html", page_title="Page Not Found"), 404

    @app.errorhandler(413)
    def payload_too_large(error):
        if wants_json():
            return {"error": "payload_too_large"}, 413
        return render_template("errors/413.html", page_title="Request Too Large"), 413

    @app.errorhandler(500)
    def server_error(error):
        app.logger.exception("Unhandled server error: %s", error)
        if wants_json():
            return {"error": "internal_error"}, 500
        return render_template("errors/500.html", page_title="Something Went Wrong"), 500

    @app.errorhandler(Exception)
    def unexpected(error):
        # Let Flask handle its own HTTP exceptions (404, 405, ...) normally.
        from werkzeug.exceptions import HTTPException

        if isinstance(error, HTTPException):
            return error

        app.logger.exception("Unexpected exception: %s", error)
        if wants_json():
            return {"error": "internal_error"}, 500
        return render_template("errors/500.html", page_title="Something Went Wrong"), 500


def _report_startup_state(logger, config) -> None:
    """Log what is and is not configured, so a misconfiguration is obvious."""
    predictor = get_predictor()

    if predictor.is_ready:
        metrics = predictor.evaluation.get("test_metrics", {})
        logger.info(
            "Model loaded: v%s, ROC-AUC %.3f, screening threshold %.3f",
            predictor.metadata.get("model_version", "?"),
            metrics.get("roc_auc", 0.0),
            predictor.screening_threshold,
        )
    else:
        logger.warning(
            "No trained model found at model/model.pkl. "
            "Run `python -m training.train` - screening will return 503 until then."
        )

    if config.llm_configured:
        logger.info("LLM narrative provider: %s", config.llm_provider)
    else:
        logger.info(
            "No LLM provider configured. Narratives will be composed from the "
            "WHO/IDF knowledge base (the app is fully functional this way)."
        )


app = create_app()


if __name__ == "__main__":
    import os
    import threading
    import webbrowser

    settings = get_config()
    print(f"\n  {APP_NAME} running at http://{settings.host}:{settings.port}\n")

    # Only open the browser once (avoids double-opening when the debugger reloads)
    if not settings.is_debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        url = f"http://{settings.host}:{settings.port}"
        threading.Timer(1.25, lambda: webbrowser.open(url)).start()

    app.run(host=settings.host, port=settings.port, debug=settings.is_debug)
