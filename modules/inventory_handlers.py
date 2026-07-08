"""Inventory and list management handlers for food items."""

import base64
import json
import logging
from datetime import datetime, timedelta

from flask import jsonify, request

import modules.core as core


def add_item_to_list(master_list_name, slave_list_name):
    """Add an item from master list to slave list."""
    try:
        item_name = request.json.get("itemName")
        if not item_name:
            return jsonify({"error": "Missing itemName"}), 400
        item_name = item_name.lower()
        master_data = core.get_data_from_json("ItemsList", master_list_name)
        if not isinstance(master_data, dict):
            return (
                jsonify({"error": f"Invalid data format in {master_list_name}.json"}),
                500,
            )

        for category, items in master_data.items():
            for item in items:
                if item["Name"].lower() == item_name:
                    slave_data = core.get_data_from_json("ItemsList", slave_list_name)
                    if category in slave_data:
                        slave_data[category].append(item)
                    else:
                        slave_data[category] = [item]
                    response = {
                        "Food": slave_data.get("Food", []),
                        "Not_Food": slave_data.get("Not_Food", []),
                    }
                    core.save_data_to_cloud_storage(
                        "ItemsList", slave_list_name, response
                    )
                    return (
                        jsonify(
                            {
                                "message": f"Item '{item_name}' added to {slave_list_name} successfully"
                            }
                        ),
                        200,
                    )
        return (
            jsonify(
                {"error": f"Item '{item_name}' not found in {master_list_name}.json"}
            ),
            404,
        )
    except Exception as exc:
        return jsonify({"error": "An internal error occurred."}), 500


def move_to_food():
    """Move an item from Not_Food category to Food category."""
    try:
        item_name = request.json.get("itemName")
        if not item_name:
            return jsonify({"error": "Item name is missing"}), 400

        nonexpired_data = core.get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(nonexpired_data, tuple):
            return (
                jsonify(
                    {"error": f"Failed to retrieve master data: {nonexpired_data[0]}"}
                ),
                500,
            )
        if isinstance(nonexpired_data, dict) and "error" in nonexpired_data:
            return (
                jsonify(
                    {
                        "error": f"Failed to retrieve master data: {nonexpired_data['error']}"
                    }
                ),
                500,
            )

        expired_data = core.get_data_from_json("ItemsList", "master_expired")
        if isinstance(expired_data, tuple):
            return (
                jsonify(
                    {"error": f"Failed to retrieve expired data: {expired_data[0]}"}
                ),
                500,
            )
        if isinstance(expired_data, dict) and "error" in expired_data:
            return (
                jsonify(
                    {
                        "error": f"Failed to retrieve expired data: {expired_data['error']}"
                    }
                ),
                500,
            )

        result_data = core.get_data_from_json("ItemsList", "result")
        if isinstance(result_data, tuple):
            return (
                jsonify({"error": f"Failed to retrieve result data: {result_data[0]}"}),
                500,
            )
        if isinstance(result_data, dict) and "error" in result_data:
            return (
                jsonify(
                    {"error": f"Failed to retrieve result data: {result_data['error']}"}
                ),
                500,
            )

        for item in nonexpired_data.get("Not_Food", []):
            if item.get("Name") == item_name:
                nonexpired_data.setdefault("Food", []).append(item)
                nonexpired_data["Not_Food"].remove(item)
                break

        if not isinstance(expired_data, dict):
            return jsonify({"error": "Invalid data format for expired_data"}), 500

        for category, items in expired_data.items():
            if not isinstance(items, list):
                return (
                    jsonify(
                        {
                            "error": f"Invalid data format in expired_data for category: {category}"
                        }
                    ),
                    500,
                )
            for item in items:
                if item.get("Name") == item_name and category != "Food":
                    nonexpired_data.setdefault("Food", []).append(item)
                    expired_data[category].remove(item)
                    break

        core.save_data_to_cloud_storage(
            "ItemsList", "master_nonexpired", nonexpired_data
        )
        core.save_data_to_cloud_storage("ItemsList", "master_expired", expired_data)

        moved_item = None
        for item in nonexpired_data.get("Food", []):
            if item.get("Name") == item_name:
                moved_item = item
                break

        if moved_item:
            result_data.setdefault("Food", []).append(moved_item)
            for category in result_data:
                if category != "Food":
                    result_data[category] = [
                        item
                        for item in result_data.get(category, [])
                        if item.get("Name") != item_name
                    ]
            core.save_data_to_cloud_storage("ItemsList", "result", result_data)

        return jsonify({"message": "Item moved to Food successfully"}), 200
    except Exception as exc:
        logging.error(f"Error in move_to_food: {str(exc)}")
        return jsonify({"error": "An internal error occurred."}), 500


def update_item_name():
    """Update the name of an item in inventory."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        old_name = data.get("itemName") or data.get("oldName")
        new_name = data.get("newName")

        if not old_name or not new_name:
            return jsonify({"error": "itemName/oldName and newName are required"}), 400

        nonexpired_content = core.get_data_from_json("ItemsList", "master_nonexpired")
        result_content = core.get_data_from_json("ItemsList", "result")

        if isinstance(nonexpired_content, tuple):
            return (
                jsonify(
                    {
                        "error": f"Failed to get master_nonexpired: {nonexpired_content[0]}"
                    }
                ),
                500,
            )
        if isinstance(result_content, tuple):
            return jsonify({"error": f"Failed to get result: {result_content[0]}"}), 500

        if isinstance(nonexpired_content, str):
            nonexpired_content = json.loads(nonexpired_content)
        if isinstance(result_content, str):
            result_content = json.loads(result_content)

        for category_key in ["Food", "Not_Food"]:
            if category_key in nonexpired_content:
                for item in nonexpired_content[category_key]:
                    if item.get("Name") == old_name:
                        item["Name"] = new_name
                        break

        core.save_data_to_cloud_storage(
            "ItemsList", "master_nonexpired", nonexpired_content
        )

        for category_key in ["Food", "Not_Food"]:
            if category_key in result_content:
                for item in result_content[category_key]:
                    if item.get("Name") == old_name:
                        item["Name"] = new_name
                        break

        core.save_data_to_cloud_storage("ItemsList", "result", result_content)
        return jsonify({"message": "Item name updated successfully"}), 200
    except Exception as exc:
        logging.error(f"Error in update_item_name: {str(exc)}")
        return jsonify({"error": "An internal error occurred."}), 500


def add_custom_item():
    """Add a custom item to the shopping list."""
    try:
        data = core.get_data_from_json("ItemsList", "shopping_list")
        request_data = request.get_json()
        item_name = request_data.get("item_name", "").lower()
        item_price = request_data.get("item_price", "$0.0")
        item_date = request_data.get("item_date", "1/1/2026")

        if not item_name:
            return jsonify({"error": "item_name is required"}), 400

        default_item = {
            "Name": "TestFNE",
            "Price": "$0.0",
            "Date": "8/1/2016",
            "Image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTvV8GjIu4AF9h-FApH1f1mkzktVXY7lhI5SDqd60AeKZtMSE6Nlpmvw7aO_Q&s",
        }

        new_item = default_item.copy()
        new_item["Name"] = item_name
        new_item["Price"] = item_price
        new_item["Date"] = item_date

        data["Food"].append(new_item)

        response = {
            "Food": data["Food"],
            "Not_Food": data["Not_Food"],
        }
        core.save_data_to_cloud_storage("ItemsList", "shopping_list", response)
        return jsonify({"message": "Custom item added successfully"})
    except Exception as exc:
        logging.error(f"Error in add_custom_item: {str(exc)}")
        return jsonify({"error": "An internal error occurred."}), 500


def delete_all_items(file_type):
    """Delete all items from a list except the test item."""
    try:
        data = core.get_data_from_json("ItemsList", file_type)
        data["Food"] = [item for item in data["Food"] if item.get("Name") == "TestFNE"]
        data["Not_Food"] = [
            item for item in data["Not_Food"] if item.get("Name") == "TestFNE"
        ]

        response = {
            "Food": data["Food"],
            "Not_Food": data["Not_Food"],
        }
        core.save_data_to_cloud_storage("ItemsList", file_type, response)
        return jsonify(
            {"message": f"{file_type.replace('_', ' ')} list deleted successfully"}
        )
    except Exception as exc:
        logging.error(f"Error in delete_all_items: {str(exc)}")
        return jsonify({"error": "An internal error occurred."}), 500


def delete_item_from_list(list_name):
    """Delete a specific item from a list."""
    try:
        json_data = core.get_data_from_json("ItemsList", list_name)
        item_name = request.json.get("itemName")
        if item_name is None:
            return jsonify({"message": "Item name is missing in the request body"}), 400

        item_found = False
        for category in json_data:
            if isinstance(json_data[category], list):
                items = json_data[category]
                for item in items:
                    if item.get("Name") == item_name:
                        items.remove(item)
                        item_found = True

        if item_found:
            response = {
                "Food": json_data.get("Food", []),
                "Not_Food": json_data.get("Not_Food", []),
            }
            core.save_data_to_cloud_storage("ItemsList", list_name, response)
            return jsonify({"message": f"Item '{item_name}' deleted successfully"}), 200
        else:
            return jsonify({"message": f"Item '{item_name}' not found"}), 404
    except Exception as exc:
        return jsonify({"message": f"An error occurred: {exc}"}), 500


def get_file_response_base64(file_name):
    """Get a list file as base64 encoded data."""
    try:
        if core.bucket is None:
            raise Exception("Storage bucket is not initialized")

        user_email = core.get_user_email_from_token()
        folder_name = f"user_{user_email}/ItemsList"
        json_blob_name = f"{folder_name}/{file_name}.json"
        json_blob = core.bucket.blob(json_blob_name)

        if json_blob.exists():
            data = json_blob.download_as_bytes()
            data_base64 = base64.b64encode(data).decode("utf-8")
            return jsonify({"data": data_base64}), 200
        return jsonify({"error": "File not found"}), 404
    except Exception as exc:
        logging.error(f"Error in get_file_response_base64: {exc}")
        return jsonify({"error": "An internal error occurred."}), 500


def update_master_nonexpired_item_expiry():
    """Extend the expiry date of an item in the inventory."""
    try:
        data = request.get_json(force=True)
        item_name = data.get("item_name", "").lower()
        days_to_extend = int(data.get("days_to_extend", 0))

        if not item_name or days_to_extend <= 0:
            return jsonify({"error": "item_name and days_to_extend are required"}), 400

        master_data = core.get_data_from_json("ItemsList", "master_nonexpired")
        shopping_data = core.get_data_from_json("ItemsList", "shopping_list")

        item_found = False
        for category, items in master_data.items():
            for item in items:
                if item.get("Name", "").lower() == item_name:
                    try:
                        expiry_date = datetime.strptime(
                            item.get("Expiry_Date", "1/1/2026"), "%d/%m/%Y"
                        )
                        new_expiry_date = expiry_date + timedelta(days=days_to_extend)
                        item["Expiry_Date"] = new_expiry_date.strftime("%d/%m/%Y")
                        if "Days_Until_Expiry" in item:
                            item["Days_Until_Expiry"] = (
                                item["Days_Until_Expiry"] + days_to_extend
                            )
                        item["Status"] = "Not Expired"
                        item_found = True
                    except ValueError:
                        pass

        for category, items in shopping_data.items():
            for item in items:
                if item.get("Name", "").lower() == item_name:
                    try:
                        expiry_date = datetime.strptime(
                            item.get("Expiry_Date", "1/1/2026"), "%d/%m/%Y"
                        )
                        new_expiry_date = expiry_date + timedelta(days=days_to_extend)
                        item["Expiry_Date"] = new_expiry_date.strftime("%d/%m/%Y")
                        if "Days_Until_Expiry" in item:
                            item["Days_Until_Expiry"] = (
                                item["Days_Until_Expiry"] + days_to_extend
                            )
                        item["Status"] = "Not Expired"
                    except ValueError:
                        pass

        if not item_found:
            return jsonify({"message": "Item not found in master_nonexpired"}), 404

        core.save_data_to_cloud_storage("ItemsList", "master_nonexpired", master_data)
        core.save_data_to_cloud_storage("ItemsList", "shopping_list", shopping_data)

        return jsonify({"message": "Expiry date extended successfully"}), 200
    except Exception as exc:
        logging.error(f"Error in update_master_nonexpired_item_expiry: {str(exc)}")
        return jsonify({"error": "An internal error occurred."}), 500


def check_frequency():
    """Check item purchase frequency based on schedule."""
    try:
        if not request.json or "condition" not in request.json:
            return (
                jsonify({"error": "Invalid input. Please provide a 'condition'."}),
                400,
            )

        choice = request.json.get("condition", "").lower()
        current_date = datetime.now()

        execute_script = False
        if choice == "biweekly":
            execute_script = current_date.day % 14 == 0
        elif choice == "monthly":
            total_days_in_month = (
                current_date.replace(month=current_date.month % 12 + 1, day=1)
                - timedelta(days=1)
            ).day
            execute_script = current_date.day == total_days_in_month
        elif choice == "today":
            execute_script = current_date.day == current_date.day
        else:
            return (
                jsonify(
                    {
                        "error": "Invalid condition. Use 'biweekly', 'monthly', or 'today'."
                    }
                ),
                400,
            )

        if not execute_script:
            return jsonify({"message": "The script will not run at this time."}), 200

        item_frequency_data = core.get_data_from_json("ItemsList", "item_frequency")

        if isinstance(item_frequency_data, tuple):
            return (
                jsonify(
                    {"error": item_frequency_data[0].get("error", "Unknown error")}
                ),
                item_frequency_data[1],
            )
        if isinstance(item_frequency_data, dict) and "error" in item_frequency_data:
            return jsonify({"error": item_frequency_data["error"]}), 500
        if isinstance(item_frequency_data, str):
            item_frequency_data = json.loads(item_frequency_data)

        item_frequency = {}
        for item in item_frequency_data.get("Food", []):
            item_name = item.get("Name")
            if item_name:
                item_frequency[item_name] = item_frequency.get(item_name, 0) + 1

        if not item_frequency:
            return jsonify({"error": "No valid item data found"}), 400

        sorted_item_frequency = dict(
            sorted(item_frequency.items(), key=lambda x: x[1], reverse=True)
        )

        core.save_data_to_cloud_storage(
            "ItemsList", "item_frequency_sorted", sorted_item_frequency
        )
        core.save_data_to_cloud_storage("ItemsList", "item_frequency", {"Food": []})

        return (
            jsonify(
                {
                    "message": "Item frequency has been saved to item_frequency_sorted.json.",
                    "sorted_item_frequency": sorted_item_frequency,
                }
            ),
            200,
        )
    except Exception as exc:
        logging.error(f"Error in check_frequency: {str(exc)}")
        return jsonify({"error": "An internal error occurred."}), 500
