"""Unit tests for facerec.auth (flask + hmac only)."""

from flask import Flask

from facerec.auth import require_api_key


def _make_app():
    app = Flask(__name__)
    app.config["API_KEY"] = "k"

    @app.route("/protected")
    @require_api_key
    def protected():
        return "ok"

    return app


def test_missing_api_key_header_returns_401():
    client = _make_app().test_client()
    response = client.get("/protected")
    assert response.status_code == 401


def test_wrong_api_key_header_returns_401():
    client = _make_app().test_client()
    response = client.get("/protected", headers={"x-api-key": "wrong"})
    assert response.status_code == 401


def test_correct_api_key_header_returns_200():
    client = _make_app().test_client()
    response = client.get("/protected", headers={"x-api-key": "k"})
    assert response.status_code == 200
    assert response.data == b"ok"
