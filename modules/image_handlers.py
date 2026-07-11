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

# Ensure Tesseract is found on Windows even if not on PATH
_TESSERACT_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESSERACT_WIN_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_WIN_PATH

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
            if not initialize_user_data_if_needed(user_email):
                raise Exception("Failed to initialize user inventory data")
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
                expiry_lookup = read_items_expiry()
                logging.info("Successfully loaded reference data files")
            except Exception as exc:
                logging.error(f"Error reading reference files: {exc}")
                return jsonify({"error": f"Failed to load reference data: {exc}"}), 500

            if filename != "dummy.jpg":
                try:
                    text = process_image(file_path)
                    logging.info(f"OCR text extracted: {len(text)} characters")

                    result = process_text(
                        text, kitchen_items, nonfood_items, irrelevant_names, expiry_lookup
                    )
                    if result is not None:
                        remove_duplicates_result(result)

                        # Accumulate purchase history: merge new items into
                        # the existing result.json without changing `result`
                        # (temp_data.json must only contain the current scan's
                        # items so master_nonexpired isn't flooded with history).
                        existing = core.get_data_from_json("ItemsList", "result")
                        if (
                            isinstance(existing, dict)
                            and "Food" in existing
                            and not isinstance(existing, tuple)
                        ):
                            for category in ("Food", "Not_Food"):
                                existing.setdefault(category, [])
                                existing[category].extend(result.get(category, []))
                            remove_duplicates_result(existing)
                            to_save = existing
                        else:
                            to_save = result

                        save_response = core.save_data_to_cloud_storage(
                            "ItemsList", "result", to_save
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

                    processing_response = process_json_files_folder(temp_dir)
                    if isinstance(processing_response, tuple):
                        _, status_code = processing_response
                    else:
                        status_code = getattr(processing_response, "status_code", 200)
                    if status_code >= 400:
                        logging.error("JSON file processing failed")
                        return processing_response
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
    """Extract text from an image using OCR with robust multi-pass preprocessing."""

    def normalize_ocr_text(text):
        """Normalize OCR text by removing control characters and noisy whitespace."""
        if not text:
            return ""

        cleaned_chars = []
        for ch in text:
            if ch == "\n" or ch == "\t" or ch.isprintable():
                cleaned_chars.append(ch)

        cleaned = "".join(cleaned_chars).replace("\t", " ")
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        return "\n".join(lines)

    def score_ocr_text(text):
        """Heuristic quality score for receipt-like OCR text."""
        if not text:
            return -1

        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return -1

        alpha_words = re.findall(r"\b[A-Za-z]{2,}\b", text)
        price_hits = re.findall(r"\$?\d+\.\d{2}\b", text)
        qty_item_hits = re.findall(r"\b\d+\s+[A-Za-z]{2,}", text)
        punctuation_noise = re.findall(r"[^\w\s\$\.,:/-]", text)

        score = 0
        score += len(lines) * 2
        score += len(alpha_words)
        score += len(price_hits) * 4
        score += len(qty_item_hits) * 2
        score -= len(punctuation_noise)
        return score

    try:
        image_cv = cv2.imread(file_path)
        if image_cv is None:
            logging.error(f"Failed to load image: {file_path}")
            return ""

        # Upscale small images to improve OCR quality for thin receipt fonts.
        height, width = image_cv.shape[:2]
        min_target_width = 1400
        if width < min_target_width:
            scale = min_target_width / float(width)
            image_cv = cv2.resize(
                image_cv,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        _, otsu = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
        )
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, morph_kernel)
        inverted = cv2.bitwise_not(otsu)

        variants = [
            ("gray", denoised),
            ("otsu", otsu),
            ("adaptive", adaptive),
            ("closed", closed),
            ("inverted", inverted),
        ]
        configs = [
            "--oem 3 --psm 6 -c preserve_interword_spaces=1",
            "--oem 3 --psm 4 -c preserve_interword_spaces=1",
            "--oem 3 --psm 11 -c preserve_interword_spaces=1",
        ]

        best_text = ""
        best_score = -1
        best_variant = ""
        best_config = ""

        for variant_name, variant_img in variants:
            for config in configs:
                raw_text = pytesseract.image_to_string(variant_img, config=config)
                cleaned = normalize_ocr_text(raw_text)
                score = score_ocr_text(cleaned)
                if score > best_score:
                    best_text = cleaned
                    best_score = score
                    best_variant = variant_name
                    best_config = config

        logging.info(
            "OCR selected variant=%s score=%s config=%s",
            best_variant,
            best_score,
            best_config,
        )
        return best_text
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract executable not found. Install Tesseract and add it to PATH.")
        return ""
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


def read_items_expiry():
    """Return a dict of item_name -> shelf_life_days from items_expiry.txt."""
    lookup = {}
    try:
        with open(core.resource_path("items_expiry.txt"), "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) >= 2 and parts[-1].strip().isdigit():
                    name = parts[0].strip().lower()
                    days = int(parts[-1].strip())
                    lookup[name] = days
    except Exception as exc:
        logging.warning("Could not load items_expiry.txt: %s", exc)
    return lookup


def _get_shelf_life(name_lower, expiry_lookup, default=7):
    """Return shelf-life days for an item, falling back to a default."""
    if name_lower in expiry_lookup:
        return expiry_lookup[name_lower]
    for key, days in expiry_lookup.items():
        if key in name_lower or name_lower in key:
            return days
    return default


def _find_best_match(name_lower, kitchen_items, nonfood_items):
    """
    Expand a partial OCR name to the closest full database entry and lock in
    its category.  Only multi-word OCR names are expanded (single words are
    too ambiguous); single words still use the normal word-score classifier.

    Strategy: the OCR words must be a *subset* of a database entry's words
    (e.g. 'dry dog' ⊂ 'dry dog food').  Among all candidates the shortest
    entry wins (fewest extra words added).

    Returns (matched_name_title, 'food'|'nonfood') or (None, None).
    """
    name_words = set(name_lower.split())
    if len(name_words) < 2:
        return None, None  # single-word names: let the normal classifier run

    best_food, best_food_extra = None, float("inf")
    best_nonfood, best_nonfood_extra = None, float("inf")

    for k in kitchen_items:
        k_words = set(k.split())
        # OCR words are all present in the database entry
        if name_words.issubset(k_words):
            extra = len(k_words) - len(name_words)
            if extra < best_food_extra:
                best_food, best_food_extra = k, extra

    for n in nonfood_items:
        n_words = set(n.split())
        if name_words.issubset(n_words):
            extra = len(n_words) - len(name_words)
            if extra < best_nonfood_extra:
                best_nonfood, best_nonfood_extra = n, extra

    if best_food is None and best_nonfood is None:
        return None, None

    # If both match, prefer the one that requires fewer extra words
    if best_food and best_nonfood:
        if best_nonfood_extra <= best_food_extra:
            return best_nonfood.title(), "nonfood"
        return best_food.title(), "food"

    if best_food:
        return best_food.title(), "food"
    return best_nonfood.title(), "nonfood"


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


def process_text(text, kitchen_items, nonfood_items, irrelevant_names, expiry_lookup=None):
    """Process OCR text into structured inventory data."""
    if expiry_lookup is None:
        expiry_lookup = {}

    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    today_str = today.strftime("%d/%m/%Y")

    # Receipt noise: skip lines containing these keywords
    _NOISE_WORDS = {
        "total", "subtotal", "sub total", "grand total", "balance",
        "tax", "gst", "hst", "pst", "vat", "discount", "savings",
        "change", "cash", "card", "visa", "mastercard", "amex",
        "debit", "credit", "payment", "paid", "amount due",
        "receipt", "invoice", "order", "transaction",
        "store", "supermarket", "market", "grocery", "tel", "phone", "www", ".com",
        "cashier", "register", "terminal", "approved", "auth",
        "thank you", "member", "loyalty", "points", "reward",
    }

    lines = text.strip().split("\n")
    lines = [row for row in lines if row.strip() != ""]
    data_list = []

    for line in lines:
        # Must contain at least one letter
        if not any(char.isalpha() for char in line):
            continue

        original_line = line
        line_lower = line.lower()

        # Skip receipt noise lines
        if any(noise in line_lower for noise in _NOISE_WORDS):
            continue

        # Skip explicitly irrelevant names
        if any(word in line_lower for word in irrelevant_names):
            continue

        # Skip lines that look like dates (DD/MM/YYYY or MM-DD-YYYY)
        if re.search(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", line):
            continue

        # Skip lines that are mostly a time stamp (HH:MM) with few letters
        if re.search(r"\b\d{1,2}:\d{2}\b", line) and len(re.findall(r"[a-zA-Z]", line)) < 4:
            continue

        # Clean: keep letters, digits, decimal points, dollar signs
        line = re.sub(r"[^a-zA-Z0-9\s\.$]", " ", line)
        line = " ".join(line.split())

        has_price = bool(re.search(r"\$?\d+\.\d{2}\b", line))
        has_leading_qty = bool(re.match(r"^\s*\d+\s+[A-Za-z]", original_line))
        # Skip non-item headers like store names when neither price nor quantity pattern is present.
        if not has_price and not has_leading_qty:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        # Extract price
        price = "$0.0"
        price_match = re.search(r"\$?(\d+\.\d{2})", line)
        if price_match:
            price = f"${price_match.group(1)}"
            line = line.replace(price_match.group(0), "").strip()
            parts = line.split()

        # Extract leading or trailing quantity token
        quantity = "1"
        if parts and re.fullmatch(r"\d+", parts[0]) and len(parts) > 1:
            quantity = parts.pop(0)
        elif parts and re.fullmatch(r"\d+", parts[-1]):
            quantity = parts.pop()

        # Drop isolated single-char tokens that are not letters (OCR noise)
        parts = [pt for pt in parts if len(pt) > 1 or pt.isalpha()]

        name = " ".join(parts).title()
        # Keep only alphabetic characters and spaces; remove digits, punctuation, etc.
        name = re.sub(r"[^a-zA-Z\s]", "", name).strip()
        name = " ".join(name.split())  # collapse multiple spaces
        if not name or len(name) < 3:
            continue

        shelf_life = _get_shelf_life(name.lower(), expiry_lookup)
        expiry_date = today + timedelta(days=shelf_life)
        expiry_str = expiry_date.strftime("%d/%m/%Y")

        data_list.append(
            {
                "Name": name,
                "Price": price,
                "Date": today_str,
                "Expiry_Date": expiry_str,
                "Status": "Not Expired",
                "Days_Until_Expiry": shelf_life,
                "Quantity": quantity,
            }
        )

    # --- Classification ---
    # Pre-build word sets for fast lookup
    food_words = set()
    for k in kitchen_items:
        food_words.update(w for w in k.split() if len(w) > 2)

    nonfood_words = set()
    for n in nonfood_items:
        nonfood_words.update(w for w in n.split() if len(w) > 2)

    def _classify(name_lower):
        name_words = set(w for w in name_lower.split() if len(w) > 2)
        # 1. Direct substring: kitchen/nonfood phrase inside item name
        # 2. Reverse substring: item name inside a kitchen/nonfood phrase
        # 3. Word-level overlap with reference word sets
        food_phrase = any(
            k in name_lower or name_lower in k
            for k in kitchen_items
        )
        nonfood_phrase = any(
            n in name_lower or name_lower in n
            for n in nonfood_items
        )
        food_score = len(name_words & food_words)
        nonfood_score = len(name_words & nonfood_words)

        if food_phrase and not nonfood_phrase:
            return "food"
        if nonfood_phrase and not food_phrase:
            return "nonfood"
        if food_phrase and nonfood_phrase:
            # Both match: trust phrase match over word score
            return "food" if food_score >= nonfood_score else "nonfood"
        # No phrase match – fall back to word-level scores
        if food_score > nonfood_score:
            return "food"
        if nonfood_score > food_score:
            return "nonfood"
        # Truly unrecognised: default to food (grocery store context)
        return "food"

    food_list = []
    not_food_list = []

    for item in data_list:
        name_lower = item["Name"].lower()

        # 1. Try word-subset expansion against the databases first.
        #    This normalises partial OCR names ("dry dog" → "Dry Dog Food")
        #    and locks in the correct category with high confidence.
        matched_name, matched_cat = _find_best_match(
            name_lower, kitchen_items, nonfood_items
        )

        if matched_name:
            item["Name"] = matched_name
            category = matched_cat
        else:
            # 2. Fall back to the score-based classifier.
            category = _classify(name_lower)

        if category == "food":
            food_list.append(item)
        else:
            not_food_list.append(item)

    return {
        "Food": food_list,
        "Not_Food": not_food_list,
    }
