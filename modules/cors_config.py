"""CORS configuration for the Flask app."""

import os

from flask import Response, jsonify, request
from flask_cors import CORS

DEFAULT_FRONTEND_ORIGIN = os.environ.get(
    "FRONTEND_APP_URL", "https://my-grocery-home.uc.r.appspot.com"
)

CORS_ORIGINS = [
    DEFAULT_FRONTEND_ORIGIN,
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:8081",
    "http://127.0.0.1:8081",
]


def _configured_origins():
    extra = os.environ.get("CORS_ORIGINS", "").strip()
    if not extra:
        return CORS_ORIGINS
    dynamic = [origin.strip() for origin in extra.split(",") if origin.strip()]
    return list(dict.fromkeys(CORS_ORIGINS + dynamic))


def _is_allowed_origin(origin):
    return bool(origin) and origin in _configured_origins()


def _add_cors_headers(response):
    origin = request.headers.get("Origin")
    if _is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = (
            request.headers.get("Access-Control-Request-Headers")
            or "Content-Type, Authorization"
        )
        response.headers["Access-Control-Allow-Methods"] = (
            request.headers.get("Access-Control-Request-Method")
            or "GET, POST, PUT, DELETE, OPTIONS"
        )
        response.headers["Access-Control-Max-Age"] = "600"
        response.headers["Vary"] = "Origin"
    return response


def handle_preflight() -> Response | None:
    if request.method == "OPTIONS" and request.path.startswith("/api/"):
        response = jsonify({"status": "ok"})
        response.status_code = 200
        return _add_cors_headers(response)
    return None


def append_cors_headers(response: Response) -> Response:
    return _add_cors_headers(response)


def configure_cors(app):
    """Attach CORS handling to a Flask app instance."""
    CORS(
        app,
        supports_credentials=True,
        resources={
            r"/api/*": {
                "origins": _configured_origins(),
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
            }
        },
    )

    app.before_request(handle_preflight)
    app.after_request(append_cors_headers)
