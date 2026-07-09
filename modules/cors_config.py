"""CORS configuration for the Flask app."""

from flask import Response, jsonify, request
from flask_cors import CORS

CORS_ORIGINS = [
    "https://my-grocery-home.uc.r.appspot.com",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:8081",
    "http://127.0.0.1:8081",
]


def _is_allowed_origin(origin):
    return bool(origin) and origin in CORS_ORIGINS


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
                "origins": CORS_ORIGINS,
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
            }
        },
    )

    app.before_request(handle_preflight)
    app.after_request(append_cors_headers)
