import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from functools import wraps

from flask import jsonify, request
from firebase_admin import auth
from google.resumable_media.common import InvalidResponse

bucket = None
bucket_name = None
storage_client = None
openai_client = None


def resource_path(relative_path):
    """Return an absolute path that works in development and in PyInstaller builds."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def sanitize_email(email: str) -> str:
    """Sanitize email values for storage object names."""
    return re.sub(r'[^a-zA-Z0-9\-_]', '_', email)


def authenticate_user_function(f):
    """Decorator to authenticate Firebase ID tokens for protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            auth_header = request.headers.get('Authorization')
            id_token = None

            if auth_header and auth_header.startswith('Bearer '):
                id_token = auth_header.split('Bearer ')[1]
            else:
                data = request.get_json(silent=True) or {}
                id_token = data.get('idToken') or request.args.get('idToken')

            if not id_token:
                logging.error("No token provided")
                return jsonify({'error': 'No token provided'}), 401

            decoded_token = auth.verify_id_token(id_token, clock_skew_seconds=60)
            request.decoded_token = decoded_token
            logging.info(f"Successfully authenticated user: {decoded_token.get('email', 'unknown')}")
            return f(*args, **kwargs)
        except auth.InvalidIdTokenError as e:
            logging.error(f"Invalid ID token: {type(e).__name__}: {str(e)}")
            return jsonify({'error': 'Invalid token', 'details': str(e)}), 401
        except auth.ExpiredIdTokenError as e:
            logging.error(f"Expired ID token: {str(e)}")
            return jsonify({'error': 'Token expired'}), 401
        except Exception as e:
            logging.error(f"Authentication error: {str(e)}")
            return jsonify({'error': 'Authentication failed'}), 401

    return decorated_function


def get_user_email_from_token():
    """Extract the authenticated user's sanitized email from the request token."""
    try:
        decoded_token = getattr(request, "decoded_token", None)

        if not decoded_token:
            auth_header = request.headers.get('Authorization')
            id_token = None
            if auth_header and auth_header.startswith('Bearer '):
                id_token = auth_header.split('Bearer ')[1]
            else:
                data = request.get_json(silent=True) or {}
                id_token = data.get('idToken') or request.args.get('idToken')

            if not id_token:
                raise Exception("Authorization token missing")

            decoded_token = auth.verify_id_token(id_token, clock_skew_seconds=60)

        email = decoded_token.get('email')
        if not email:
            raise Exception("Email not found in token")

        return sanitize_email(email)
    except Exception as exc:
        logging.error(f"Failed to get user email from token: {exc}")
        return None


def get_data_from_json(folder_name, file_name):
    """Retrieve JSON content for the authenticated user from Cloud Storage."""
    try:
        if bucket is None:
            raise Exception("Storage bucket is not initialized")

        user_email = get_user_email_from_token()
        if not user_email:
            raise Exception("Unable to determine user from token")

        json_blob_name = f"user_{user_email}/{folder_name}/{file_name}.json"
        blob = bucket.blob(json_blob_name)

        if blob.exists():
            content = blob.download_as_text()
            return json.loads(content)

        logging.warning(f"File not found: {json_blob_name}")

        if folder_name.startswith("ChatGPT"):
            default_data = {
                'last_generated': datetime.min.isoformat(),
                'jokes': []
            } if file_name == "Joke" else []
            blob.upload_from_string(json.dumps(default_data, indent=4), if_generation_match=0)
            return default_data

        if file_name in ("master_nonexpired", "master_expired", "shopping_list", "result"):
            return {"Food": [], "Not_Food": []}
        if file_name == "item_frequency":
            return {"Food": []}
        if file_name == "Joke":
            return {'last_generated': datetime.min.isoformat(), 'jokes': []}

        return {"error": "File not found"}, 404
    except json.JSONDecodeError as exc:
        logging.error(f"JSON decode error for {file_name}: {exc}")
        return {"error": "Invalid JSON format"}, 500
    except Exception as exc:
        logging.error(f"Error getting data from {file_name}: {exc}")
        return {"error": str(exc)}, 500


def save_data_to_cloud_storage(folder_name, file_name, data, max_retries=5):
    """Save JSON content for the authenticated user to Cloud Storage."""
    try:
        if bucket is None:
            raise Exception("Storage bucket is not initialized")

        user_email = get_user_email_from_token()
        if not user_email:
            raise Exception("Unable to determine user from token")

        json_blob_name = f"user_{user_email}/{folder_name}/{file_name}.json"
        blob = bucket.blob(json_blob_name)
        attempt = 0
        backoff = 1

        while attempt < max_retries:
            try:
                blob.upload_from_string(json.dumps(data, indent=4), if_generation_match=0)
                logging.info(f"Data saved to {json_blob_name}")
                return {"message": "Data saved successfully"}, 200
            except InvalidResponse as exc:
                if exc.response.status_code == 429:
                    logging.warning(f"Rate limit exceeded. Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue
                logging.exception("Exception occurred while saving data to cloud storage")
                return {"error": str(exc)}, 500
            except Exception as exc:
                logging.exception("Unexpected exception occurred while saving data to cloud storage")
                return {"error": str(exc)}, 500

        logging.error("Max retries exceeded. Failed to save the file.")
        return {"error": "Max retries exceeded. Failed to save the file."}, 500
    except Exception as exc:
        logging.error(f"Error saving data to {file_name}: {exc}")
        return {"error": str(exc)}, 500
