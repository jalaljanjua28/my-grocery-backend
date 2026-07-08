"""Image processing and OCR-related handlers for receipt scanning."""

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta

import cv2
import pytesseract
from flask import jsonify, request

import modules.core as core
from modules.data_processing_handlers import (
    create_master_expired_file,
    process_json_files_folder,
    remove_duplicates_result,
)
from modules.user_handlers import initialize_user_data_if_needed


def compare_image_function():
    """Extract OCR text from an uploaded image."""
    try:
        if "file" not in request.files:
            return jsonify({"message": "No file provided"}), 400

        file = request.files["file"]
        ocr_text = ""

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = file.filename
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            if filename != "dummy.jpg":
                raw_text = process_image(file_path)
                ocr_lines = raw_text.split("\n")
                cleaned_lines = [line.strip() for line in ocr_lines if line.strip()]
                ocr_text = "\n".join(cleaned_lines)

        return jsonify({"message": "Image processed successfully", "ocrText": ocr_text})
    except Exception as exc:
        logging.error(f"Error in compare_image_function: {exc}")
        return jsonify({"error": "An internal error occurred."}), 500


def main_function():
    """Process an uploaded receipt image and update the inventory files."""
    try:
        blob = None
        result = None

        if "file" not in request.files:
            return jsonify({"message": "No file provided"}), 400

        file = request.files["file"]

        try:
            user_email = core.get_user_email_from_token()
            logging.info(f"Processing request for user: {user_email}")
            initialize_user_data_if_needed(user_email)
        except Exception as exc:
            logging.error(f"Error initializing user data: {exc}")
            return jsonify({"error": f"Failed to initialize user data: {exc}"}), 500

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = file.filename
            file_path = os.path.join(temp_dir, filename)

            try:
                file.save(file_path)
                blob = core.storage_client.bucket(core.bucket_name).blob(filename)
                blob.upload_from_filename(file_path)
                logging.info(f"File {filename} uploaded to GCS successfully")
            except Exception as exc:
                logging.error(f"Error uploading file to GCS: {exc}")
                return jsonify({"error": f"Failed to upload file: {exc}"}), 500

            try:
                kitchen_items = read_kitchen_eatables()
                nonfood_items = nonfood_items_list()
                irrelevant_names = irrelevant_names_list()
                logging.info("Successfully loaded reference data files")
            except Exception as exc:
                logging.error(f"Error reading reference files: {exc}")
                return jsonify({"error": f"Failed to load reference data: {exc}"}), 500

            if filename != "dummy.jpg":
                try:
                    text = process_image(file_path)
                    logging.info(f"OCR text extracted: {len(text)} characters")

                    result = process_text(
                        text, kitchen_items, nonfood_items, irrelevant_names
                    )
                    if result is not None:
                        remove_duplicates_result(result)
                        save_response = core.save_data_to_cloud_storage(
                            "ItemsList", "result", result
                        )
                        if isinstance(save_response, tuple) and save_response[1] != 200:
                            logging.error(f"Error saving result: {save_response[0]}")
                            return (
                                jsonify({"error": "Failed to save processed data"}),
                                500,
                            )
                        logging.info("Result data saved successfully")

                    temp_file_path = os.path.join(temp_dir, "temp_data.json")
                    with open(temp_file_path, "w") as json_file:
                        json.dump(result, json_file, indent=4)

                    process_json_files_folder(temp_dir)
                    logging.info("JSON files processed successfully")
                except Exception as exc:
                    logging.error(f"Error processing image/text: {exc}")
                    return jsonify({"error": f"Failed to process image: {exc}"}), 500

            try:
                data_nonexpired = core.get_data_from_json(
                    "ItemsList", "master_nonexpired"
                )
                if isinstance(data_nonexpired, tuple):
                    logging.error(
                        f"Error getting master_nonexpired: {data_nonexpired[0]}"
                    )
                    data_nonexpired = {"Food": [], "Not_Food": []}

                if isinstance(data_nonexpired, dict) and "error" in data_nonexpired:
                    logging.error(
                        f"Error in master_nonexpired data: {data_nonexpired['error']}"
                    )
                    data_nonexpired = {"Food": [], "Not_Food": []}

                create_master_expired_file(data_nonexpired)
                logging.info("Master expired file created successfully")

                save_response = core.save_data_to_cloud_storage(
                    "ItemsList", "master_nonexpired", data_nonexpired
                )
                if isinstance(save_response, tuple) and save_response[1] != 200:
                    logging.error(f"Error saving master_nonexpired: {save_response[0]}")
                    return jsonify({"error": "Failed to save master data"}), 500
            except Exception as exc:
                logging.error(f"Error processing master files: {exc}")
                return jsonify({"error": f"Failed to process master files: {exc}"}), 500

            try:
                if blob:
                    blob.reload()
                    if blob.exists():
                        blob.delete()
                        logging.info(f"Temporary file {filename} deleted from GCS")
            except Exception as exc:
                logging.warning(f"Error deleting temporary file: {exc}")

        return jsonify({"message": "File uploaded and processed successfully"})
    except Exception as exc:
        logging.error(f"Unexpected error in main_function: {exc}")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500


def process_image(file_path):
    """Extract text from an image using OCR with improved preprocessing."""
    try:
        image_cv = cv2.imread(file_path)
        if image_cv is None:
            logging.error(f"Failed to load image: {file_path}")
            return ""

        # Convert to grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Custom config for PyTesseract (psm 4 assumes a single column of text of variable sizes)
        custom_config = r"--oem 3 --psm 4"
        text = pytesseract.image_to_string(thresh, config=custom_config)
        print(text)
        return text
    except Exception as exc:
        logging.error(f"Error processing image: {exc}")
        return ""


def read_kitchen_eatables():
    """Return the list of kitchen items from the local reference file."""
    kitchen_items = []
    with open(
        core.resource_path("Kitchen_Eatables_Database.txt"), "r", encoding="utf-8"
    ) as handle:
        for line in handle:
            kitchen_items.append(line.strip().lower())
    return kitchen_items


def nonfood_items_list():
    """Return the list of non-food items from the local reference file."""
    nonfood_items = []
    with open(core.resource_path("NonFoodItems.txt"), "r", encoding="utf-8") as handle:
        for line in handle:
            nonfood_items.append(line.strip().lower())
    return nonfood_items


def irrelevant_names_list():
    """Return irrelevant names from the local reference file."""
    irrelevant_names = []
    with open(core.resource_path("Irrelevant.txt"), "r", encoding="utf-8") as handle:
        for line in handle:
            irrelevant_names.append(line.strip().lower())
    return irrelevant_names


def add_days(date_str, days_to_add):
    """Add a number of days to a date string."""
    date_format = "%d/%m/%Y"
    date_obj = datetime.strptime(date_str, date_format)
    new_date = date_obj + timedelta(days=days_to_add)
    return new_date.strftime(date_format)


def process_text(text, kitchen_items, nonfood_items, irrelevant_names):
    """Process OCR text into structured inventory data."""
    lines = text.strip().split("\n")
    lines = [row for row in lines if row.strip() != ""]
    data_list = []

    for line in lines:
        if not any(char.isalpha() for char in line):
            continue

        if any(word in line.lower() for word in irrelevant_names):
            continue

        # Clean up line but preserve letters, digits, decimals, and dollar signs
        line = re.sub(r"[^a-zA-Z0-9\s\.$]", " ", line)
        line = " ".join(line.split())

        parts = line.split()
        if len(parts) < 2:
            continue

        # Extract price if present
        price = "$0.0"
        price_match = re.search(r"\$?(\d+\.\d{2})", line)
        if price_match:
            price = f"${price_match.group(1)}"
            line = line.replace(price_match.group(0), "").strip()
            parts = line.split()

        # Extract quantity if present
        quantity = "1"
        if parts and parts[-1].isdigit():
            quantity = parts.pop()

        name = " ".join(parts).title()
        if not name:
            continue

        data_list.append(
            {
                "Name": name,
                "Price": price,
                "Date": "1/1/2026",
                "Expiry_Date": "01/01/2026",
                "Status": "Not Expired",
                "Days_Until_Expiry": 0,
                "Quantity": quantity,
            }
        )

    # Improved classification
    food_list = []
    not_food_list = []

    for item in data_list:
        name_lower = item["Name"].lower()

        # Check if any word in the name matches known items
        is_food = any(k_item in name_lower for k_item in kitchen_items)
        is_nonfood = any(nf_item in name_lower for nf_item in nonfood_items)

        if is_food:
            food_list.append(item)
        elif is_nonfood:
            not_food_list.append(item)
        else:
            # Fallback for unclassified items
            not_food_list.append(item)

    return {
        "Food": food_list,
        "Not_Food": not_food_list,
    }
