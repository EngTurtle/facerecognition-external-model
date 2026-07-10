"""Flask application factory."""

import logging

from flask import Flask

from facerec.api import bp
from facerec.config import Config
from facerec.model import FaceModel


def _configure_logging() -> None:
    """Route logging through gunicorn's handlers when running under gunicorn."""
    gunicorn_logger = logging.getLogger("gunicorn.error")
    if gunicorn_logger.handlers:
        root_logger = logging.getLogger()
        root_logger.handlers = gunicorn_logger.handlers
        root_logger.setLevel(gunicorn_logger.level)
    else:
        logging.basicConfig(level=logging.INFO)


def create_app(config: Config | None = None) -> Flask:
    """Build and configure the Flask application."""
    _configure_logging()

    config = config or Config.from_env()

    app = Flask(__name__)
    app.config["API_KEY"] = config.api_key
    app.extensions["facerec_config"] = config
    app.extensions["face_model"] = FaceModel(config)
    app.register_blueprint(bp)

    return app
