"""API key authentication decorator for Flask views."""

import hmac
from functools import wraps

from flask import abort, current_app, request


def require_api_key(view):
    """Require a valid `x-api-key` header, checked against app.config["API_KEY"]."""

    @wraps(view)
    def decorated_function(*args, **kwargs):
        expected_key = current_app.config["API_KEY"]
        provided_key = request.headers.get("x-api-key")
        if not provided_key or not hmac.compare_digest(provided_key, expected_key):
            abort(401)
        return view(*args, **kwargs)

    return decorated_function
