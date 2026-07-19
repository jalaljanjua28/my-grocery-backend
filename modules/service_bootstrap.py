"""Service initialization for Secret Manager, Firebase, Storage, and OpenAI."""

import json
import logging
import os
import random
import time

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import DeadlineExceeded
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import secretmanager_v1, storage
from google.oauth2 import service_account

import modules.core as core


def access_secret_version(secret_client, project_id, secret_id, timeout=60):
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = secret_client.access_secret_version(
        request={"name": name}, timeout=timeout
    )
    return response.payload.data.decode("UTF-8")


def initialize_firebase(secret_client, project_id, retries=5):
    firebase_secret_id = "firebase_service_account"
    for attempt in range(retries):
        try:
            firebase_cred_data = access_secret_version(
                secret_client, project_id, firebase_secret_id
            )
            firebase_cred_dict = json.loads(firebase_cred_data)
            cred = credentials.Certificate(firebase_cred_dict)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            logging.info(
                "Firebase credentials retrieved and app initialized successfully."
            )
            return firestore.client()
        except DeadlineExceeded:
            sleep_time = (2**attempt) + random.uniform(0, 1)
            logging.warning(
                "Firebase init deadline exceeded, retrying in %.2f seconds", sleep_time
            )
            time.sleep(sleep_time)
        except Exception as exc:
            logging.error("Error initializing Firebase app: %s", exc)
            break

    # Fallback for Cloud Run/App Engine with ADC and attached service account.
    try:
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        logging.info("Firebase initialized with Application Default Credentials.")
        return firestore.client()
    except Exception as exc:
        logging.error("Firebase ADC fallback failed: %s", exc)
    return None


def initialize_storage(secret_client, project_id, bucket_name):
    try:
        service_account_secret_id = "my-credentials-json"
        service_account_key = access_secret_version(
            secret_client, project_id, service_account_secret_id
        )
        creds = service_account.Credentials.from_service_account_info(
            json.loads(service_account_key)
        )
        core.storage_client = storage.Client(credentials=creds, project=project_id)
        core.bucket_name = bucket_name
        core.bucket = core.storage_client.bucket(bucket_name)
        logging.info("Service account key retrieved successfully.")
    except Exception as exc:
        logging.warning("Secret-based storage init failed, falling back to ADC: %s", exc)
        try:
            core.storage_client = storage.Client(project=project_id)
            core.bucket_name = bucket_name
            core.bucket = core.storage_client.bucket(bucket_name)
            logging.info("Cloud Storage initialized with Application Default Credentials.")
        except Exception as adc_exc:
            logging.error("Storage ADC fallback failed: %s", adc_exc)

    try:
        app_default_secret_id = "Application-default-credentials"
        _ = access_secret_version(
            secret_client, project_id, app_default_secret_id
        ).strip()
        logging.info("Application Default Credentials retrieved successfully.")
    except Exception as exc:
        logging.error("Error retrieving application default credentials: %s", exc)


def initialize_openai(secret_client, project_id):
    existing_key = os.environ.get("OPENAI_API_KEY")
    if existing_key:
        try:
            from openai import OpenAI

            core.openai_client = OpenAI(api_key=existing_key)
            logging.info("OpenAI client initialized from environment variable.")
            return
        except Exception as exc:
            logging.error("OpenAI environment initialization failed: %s", exc)

    try:
        openai_secret_id = "OPENAI-API-KEY"
        openai_api_key = access_secret_version(
            secret_client, project_id, openai_secret_id
        )
        os.environ["OPENAI_API_KEY"] = openai_api_key
        from openai import OpenAI

        core.openai_client = OpenAI(api_key=openai_api_key)
        logging.info("OpenAI API key retrieved successfully.")
    except Exception as exc:
        logging.error(
            "Error retrieving OpenAI API key from Google Secret Manager: %s", exc
        )
        core.openai_client = None


def _sort_data_files():
    """Sort and deduplicate local data files at startup."""
    try:
        from modules.data_processing_handlers import clean_and_sort_files

        filenames = [
            core.resource_path("items_expiry.txt"),
            core.resource_path("NonFoodItems.txt"),
            core.resource_path("Kitchen_Eatables_Database.txt"),
            core.resource_path("Irrelevant.txt"),
            core.resource_path("ItemCost.txt"),
        ]
        clean_and_sort_files(filenames)
    except Exception as exc:
        logging.warning("clean_and_sort_files failed at startup: %s", exc)


def initialize_services(project_id, bucket_name):
    try:
        secret_client = secretmanager_v1.SecretManagerServiceClient()
    except DefaultCredentialsError as exc:
        logging.warning(
            "Google ADC not found; starting with local-only services: %s", exc
        )
        try:
            if not firebase_admin._apps:
                # Local fallback: token verification can work with projectId set,
                # even when full cloud credentials are not available.
                firebase_admin.initialize_app(options={"projectId": project_id})
            logging.info("Firebase initialized in local fallback mode.")
        except Exception as fb_exc:
            logging.warning("Local Firebase fallback init failed: %s", fb_exc)
        core.bucket_name = bucket_name
        core.bucket = None
        core.storage_client = None
        # OpenAI does not depend on Google ADC. Local development commonly uses
        # OPENAI_API_KEY without any Google credentials, so initialize it before
        # returning from the cloud-services fallback.
        initialize_openai(None, project_id)
        _sort_data_files()
        return {"secret_client": None, "db": None}

    db = initialize_firebase(secret_client, project_id)
    initialize_storage(secret_client, project_id, bucket_name)
    initialize_openai(secret_client, project_id)
    _sort_data_files()
    return {"secret_client": secret_client, "db": db}
