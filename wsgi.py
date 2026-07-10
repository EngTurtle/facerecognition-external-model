"""Gunicorn entrypoint: `gunicorn -c gunicorn_config.py wsgi:app`."""

from facerec.app import create_app

app = create_app()

if __name__ == "__main__":
    import os

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
