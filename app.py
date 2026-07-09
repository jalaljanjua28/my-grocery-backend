# Standard library imports
import logging
import os
import sys
import threading
import time

# Flask imports
from flask import Flask, jsonify, send_from_directory

import modules.core as core
from modules.cors_config import configure_cors
from modules.service_bootstrap import initialize_services
import modules.chatgpt_routes as chatgpt_routes
import modules.inventory_routes as inventory_routes
import modules.image_routes as image_routes
import modules.user_routes as user_routes

try:
    import webview
except ImportError:
    webview = None

# App creation
_static = core.resource_path("dist") if getattr(sys, "frozen", False) else "dist"
app = Flask(__name__, static_folder=_static, template_folder=_static)

configure_cors(app)

app.register_blueprint(chatgpt_routes.bp)
app.register_blueprint(inventory_routes.bp)
app.register_blueprint(image_routes.bp)
app.register_blueprint(user_routes.bp)

# Services
os.environ.setdefault("BUCKET_NAME", "my-grocery")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "my-grocery-home")
logging.basicConfig(level=logging.DEBUG)
service_context = initialize_services(
    os.environ["GOOGLE_CLOUD_PROJECT"], os.environ["BUCKET_NAME"]
)
db = service_context.get("db")


# Static file serving (SPA catch-all)
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    file_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


def start_flask():
    host = "0.0.0.0" if os.environ.get("K_SERVICE") else "127.0.0.1"
    app.run(
        debug=False, host=host, port=int(os.environ.get("PORT", 8081)), threaded=True
    )


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        if webview is not None:
            threading.Thread(target=start_flask, daemon=True).start()
            time.sleep(2)
            webview.create_window(
                "My Grocery Home",
                "https://my-grocery-home.uc.r.appspot.com",
                width=1200,
                height=800,
                resizable=True,
            )
            webview.start(debug=False)
        else:
            start_flask()
    else:
        if webview is not None:
            threading.Thread(target=start_flask, daemon=True).start()
            webview.create_window(
                "My Grocery Home", "https://my-grocery-home.uc.r.appspot.com"
            )
            webview.start()
        else:
            start_flask()
