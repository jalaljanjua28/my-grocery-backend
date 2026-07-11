"""Data processing helpers for inventory and OCR pipeline workflows."""

import json
import logging
import os
from datetime import datetime

from flask import jsonify

import modules.core as core


def read_json_file(file_path):
    """Read a JSON file from disk."""
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def calculate_days_until_expiry(item):
    """Calculate how many days remain until an item expires from today."""
    expiry_date = datetime.strptime(item["Expiry_Date"], "%d/%m/%Y")
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    return (expiry_date - today).days


def get_data_from_json(folder_name, file_name):
    return core.get_data_from_json(folder_name, file_name)


def save_data_to_cloud_storage(folder_name, file_name, data, max_retries=5):
    return core.save_data_to_cloud_storage(
        folder_name, file_name, data, max_retries=max_retries
    )


def resource_path(path):
    return core.resource_path(path)


def append_unique_to_master_nonexpired(
    master_nonexpired_data, data_to_append, category
):
    """Append unique items from OCR output to the master non-expired data."""
    for item_to_append in data_to_append[category]:
        item_unique = True
        for master_nonexpired_item in master_nonexpired_data[category]:
            if all(
                item_to_append[prop] == master_nonexpired_item[prop]
                for prop in ["Name", "Price", "Date", "Expiry_Date", "Status"]
            ):
                item_unique = False
                break
        if item_unique:
            days_until_expiry = calculate_days_until_expiry(item_to_append)
            item_to_append["Days_Until_Expiry"] = days_until_expiry
            master_nonexpired_data[category].append(item_to_append)
    return master_nonexpired_data


def remove_duplicates_nonexpired(master_nonexpired_data):
    """Remove duplicate items from non-expired data."""
    for category, items in master_nonexpired_data.items():
        unique_items = {}
        for item in items:
            name = item["Name"]
            price = item["Price"]
            if isinstance(price, str):
                price_str = price.replace("$", "").replace(",", "")
            else:
                price_str = str(price)
            price = float(price_str)
            if name not in unique_items:
                unique_items[name] = item
            else:
                existing_price = unique_items[name]["Price"]
                if isinstance(existing_price, str):
                    existing_price_str = existing_price.replace("$", "").replace(
                        ",", ""
                    )
                else:
                    existing_price_str = str(existing_price)
                existing_price = float(existing_price_str)
                if price < existing_price:
                    unique_items[name] = item
        master_nonexpired_data[category] = list(unique_items.values())
    return master_nonexpired_data


def remove_duplicates_result(result):
    """Remove duplicate items from the OCR result data."""
    for category, items in result.items():
        unique_items = {}
        for item in items:
            name = item["Name"].lower()
            price = item["Price"]
            if isinstance(price, str):
                price_str = price.replace("$", "").replace(",", "")
            else:
                price_str = str(price)
            price = float(price_str)
            if name not in unique_items or price < unique_items[name]["price"]:
                unique_items[name] = {"item": item, "price": price}
        result[category] = [item_data["item"] for item_data in unique_items.values()]
    return result


def remove_duplicates_expired(data_expired):
    """Remove duplicate items from expired data."""
    for category, items in data_expired.items():
        unique_items = {}
        for item in items:
            name = item["Name"]
            price = item["Price"]
            if isinstance(price, str):
                price_str = price.replace("$", "").replace(",", "")
            else:
                price_str = str(price)
            price = float(price_str)
            if name not in unique_items:
                unique_items[name] = item
            else:
                existing_price = unique_items[name]["Price"]
                if isinstance(existing_price, str):
                    existing_price_str = existing_price.replace("$", "").replace(
                        ",", ""
                    )
                else:
                    existing_price_str = str(existing_price)
                existing_price = float(existing_price_str)
                if price < existing_price:
                    unique_items[name] = item
        data_expired[category] = list(unique_items.values())
    return data_expired


def remove_items_present_in_expired_from_nonexpired(
    master_nonexpired_data, master_expired_data
):
    """Remove items from the non-expired set when they already appear in the expired set."""
    for category, expired_items in master_expired_data.items():
        nonexpired_items = master_nonexpired_data.get(category, [])
        items_to_remove = []
        for expired_item in expired_items:
            for nonexpired_item in nonexpired_items:
                if (
                    expired_item["Name"] == nonexpired_item["Name"]
                    and expired_item["Expiry_Date"] == nonexpired_item["Expiry_Date"]
                ):
                    items_to_remove.append(nonexpired_item)
                    break
        master_nonexpired_data[category] = [
            item for item in nonexpired_items if item not in items_to_remove
        ]
    return master_nonexpired_data


def process_json_files_folder(temp_dir):
    """Append data from JSON files generated during OCR processing to the master inventory."""
    try:
        master_nonexpired_data = get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(master_nonexpired_data, tuple):
            logging.error(
                f"Error getting master_nonexpired in process_json_files_folder: {master_nonexpired_data[0]}"
            )
            return jsonify({"error": "Failed to retrieve master data"}), 500
        if (
            isinstance(master_nonexpired_data, dict)
            and "error" in master_nonexpired_data
        ):
            logging.error(
                f"Error in master_nonexpired data: {master_nonexpired_data['error']}"
            )
            return jsonify({"error": "Failed to retrieve master data"}), 500

        json_files_to_append = [f for f in os.listdir(temp_dir) if f.endswith(".json")]
        data_to_append = None
        for json_file in json_files_to_append:
            json_file_path = os.path.join(temp_dir, json_file)
            try:
                data_to_append = read_json_file(json_file_path)
            except Exception as exc:
                logging.error(f"Error reading JSON file {json_file}: {exc}")
                continue

        if data_to_append is not None and os.path.isfile(
            os.path.join(temp_dir, json_files_to_append[-1])
        ):
            try:
                append_unique_to_master_nonexpired(
                    master_nonexpired_data, data_to_append, "Food"
                )
                append_unique_to_master_nonexpired(
                    master_nonexpired_data, data_to_append, "Not_Food"
                )
            except Exception as exc:
                logging.error(f"Error appending data: {exc}")
                return jsonify({"error": f"Failed to append data: {exc}"}), 500
        else:
            logging.warning("No JSON file found to process")

        save_response = save_data_to_cloud_storage(
            "ItemsList", "master_nonexpired", master_nonexpired_data
        )
        if isinstance(save_response, tuple) and save_response[1] != 200:
            logging.error(f"Error saving master_nonexpired: {save_response[0]}")
            return jsonify({"error": "Failed to save updated data"}), 500

        return jsonify({"message": "Data appended successfully"})
    except Exception as exc:
        logging.error(f"Unexpected error in process_json_files_folder: {exc}")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500


def create_master_expired_file(data_nonexpired):
    """Move expired items from the non-expired inventory to the expired inventory."""
    try:
        data_expired = get_data_from_json("ItemsList", "master_expired")
    except FileNotFoundError:
        data_expired = {"Food": [], "Not_Food": []}

    today = datetime.today().strftime("%d/%m/%Y")
    items_to_remove = []
    for category, items in data_nonexpired.items():
        for item in items:
            expiry_date_str = item["Expiry_Date"]
            expiry_date = datetime.strptime(expiry_date_str, "%d/%m/%Y")
            today_date = datetime.strptime(today, "%d/%m/%Y")
            if expiry_date < today_date:
                if not any(
                    expired_item.get("Name") == item["Name"]
                    and expired_item.get("Expiry_Date") == expiry_date_str
                    for expired_item in data_expired[category]
                ):
                    item["Status"] = "Expired"
                    data_expired[category].append(item)
                    items_to_remove.append(item)

    for category in data_nonexpired:
        data_nonexpired[category] = [
            item for item in data_nonexpired[category] if item not in items_to_remove
        ]

    data_nonexpired = remove_items_present_in_expired_from_nonexpired(
        data_nonexpired, data_expired
    )
    remove_duplicates_expired(data_expired)
    remove_duplicates_nonexpired(data_nonexpired)

    save_data_to_cloud_storage("ItemsList", "master_expired", data_expired)
    save_data_to_cloud_storage("ItemsList", "master_nonexpired", data_nonexpired)
    return jsonify({"message": "Expired item list created successfully"})


def clean_and_sort_files(filenames):
    """Clean and sort reference lists from the local data files."""
    import chardet

    for filename in filenames:
        full_path = resource_path(filename)
        items = {}
        with open(full_path, "rb") as raw_file:
            raw_data = raw_file.read()
            detected = chardet.detect(raw_data)
            encoding = detected["encoding"] or "utf-8"
        with open(full_path, "r", encoding=encoding, errors="ignore") as handle:
            for line in handle:
                parts = line.strip().lower().split(",")
                name = parts[0].strip()
                days = parts[-1].strip() if len(parts) > 1 else ""
                if name not in items or (days.isdigit() and int(days) < items[name]):
                    items[name] = int(days) if days.isdigit() else ""
        with open(full_path, "w", encoding="utf-8") as handle:
            for name, days in sorted(items.items()):
                handle.write(f"{name},{days}\n" if days != "" else f"{name}\n")
    return True
