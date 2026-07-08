"""User setup, account initialization, and related utility handlers."""

import json
import logging
from datetime import datetime

from flask import jsonify, request
from firebase_admin import firestore

import modules.core as core


def create_missing_chatgpt_files_function():
    """Create missing ChatGPT storage files for the current user."""
    try:
        user_email = core.get_user_email_from_token()
        logging.info(f"Creating missing ChatGPT files for user: {user_email}")

        chatgpt_files = {
            f"user_{user_email}/ChatGPT/HomePage/food_handling_advice.json": [],
            f"user_{user_email}/ChatGPT/HomePage/Food_Waste_Reduction_Suggestions.json": [],
            f"user_{user_email}/ChatGPT/HomePage/Ethical_Eating_Suggestions.json": [],
            f"user_{user_email}/ChatGPT/HomePage/generated_fun_facts.json": [],
            f"user_{user_email}/ChatGPT/HomePage/Cooking_Tips.json": [],
            f"user_{user_email}/ChatGPT/HomePage/Current_Trends.json": [],
            f"user_{user_email}/ChatGPT/HomePage/Mood_Changer.json": [],
            f"user_{user_email}/ChatGPT/HomePage/Joke.json": {
                "last_generated": datetime.min.isoformat(),
                "jokes": [],
            },
            f"user_{user_email}/ChatGPT/Health/generated_nutritional_advice.json": [],
            f"user_{user_email}/ChatGPT/Health/allergy_information.json": [],
            f"user_{user_email}/ChatGPT/Health/Healthy_alternatives.json": [],
            f"user_{user_email}/ChatGPT/Health/healthy_eating_advice.json": [],
            f"user_{user_email}/ChatGPT/Health/Health_Advice.json": [],
            f"user_{user_email}/ChatGPT/Health/healthy_usage.json": [],
            f"user_{user_email}/ChatGPT/Health/Nutritional_Analysis.json": [],
            f"user_{user_email}/ChatGPT/Health/health_incompatibility_information_all.json": [],
            f"user_{user_email}/ChatGPT/Recipe/User_Defined_Dish.json": [],
            f"user_{user_email}/ChatGPT/Recipe/Fusion_Cuisine_Suggestions.json": [],
            f"user_{user_email}/ChatGPT/Recipe/Unique_Recipes.json": [],
            f"user_{user_email}/ChatGPT/Recipe/generated_recipes.json": [],
            f"user_{user_email}/ChatGPT/Recipe/diet_schedule.json": [],
        }

        created_files = []
        skipped_files = []
        failed_files = []

        for file_path, default_data in chatgpt_files.items():
            try:
                blob = core.bucket.blob(file_path)
                if blob.exists():
                    skipped_files.append(file_path)
                    continue

                blob.upload_from_string(
                    json.dumps(default_data, indent=4), if_generation_match=0
                )
                created_files.append(file_path)
            except Exception as exc:
                failed_files.append(
                    {"file": file_path, "error": "Internal creation error"}
                )
                logging.error(f"Failed to create {file_path}: {exc}")

        return (
            jsonify(
                {
                    "message": "ChatGPT files creation completed",
                    "user_email": user_email,
                    "created_files": len(created_files),
                    "skipped_files": len(skipped_files),
                    "failed_files": len(failed_files),
                    "details": {
                        "created": created_files,
                        "skipped": skipped_files,
                        "failed": failed_files,
                    },
                }
            ),
            200,
        )
    except Exception as exc:
        logging.error(f"Error creating ChatGPT files: {exc}")
        return jsonify({"error": "An internal error occurred."}), 500


def check_user_files_function():
    """Inspect the current user's files and report missing or misplaced data."""
    try:
        user_email = core.get_user_email_from_token()
        logging.info(f"Checking files for user: {user_email}")

        prefix = f"user_{user_email}/"
        blobs = core.bucket.list_blobs(prefix=prefix)
        existing_files = [blob.name for blob in blobs]

        itemslist_files = [f for f in existing_files if "/ItemsList/" in f]
        chatgpt_homepage_files = [
            f for f in existing_files if "/ChatGPT/HomePage/" in f
        ]
        chatgpt_health_files = [f for f in existing_files if "/ChatGPT/Health/" in f]
        chatgpt_recipe_files = [f for f in existing_files if "/ChatGPT/Recipe/" in f]

        improper_files = []
        user_info_files = []
        for file_path in existing_files:
            if any(
                folder in file_path
                for folder in [
                    "/ItemsList/",
                    "/ChatGPT/HomePage/",
                    "/ChatGPT/Health/",
                    "/ChatGPT/Recipe/",
                ]
            ):
                continue
            if ".user_info.json" in file_path:
                user_info_files.append(file_path)
            else:
                improper_files.append(file_path)

        required_files = [
            f"user_{user_email}/ItemsList/master_nonexpired.json",
            f"user_{user_email}/ItemsList/master_expired.json",
            f"user_{user_email}/ItemsList/shopping_list.json",
            f"user_{user_email}/ItemsList/result.json",
            f"user_{user_email}/ItemsList/item_frequency.json",
            f"user_{user_email}/ChatGPT/HomePage/food_handling_advice.json",
            f"user_{user_email}/ChatGPT/HomePage/Food_Waste_Reduction_Suggestions.json",
            f"user_{user_email}/ChatGPT/HomePage/Ethical_Eating_Suggestions.json",
            f"user_{user_email}/ChatGPT/HomePage/generated_fun_facts.json",
            f"user_{user_email}/ChatGPT/HomePage/Cooking_Tips.json",
            f"user_{user_email}/ChatGPT/HomePage/Current_Trends.json",
            f"user_{user_email}/ChatGPT/HomePage/Mood_Changer.json",
            f"user_{user_email}/ChatGPT/HomePage/Joke.json",
            f"user_{user_email}/ChatGPT/Health/generated_nutritional_advice.json",
            f"user_{user_email}/ChatGPT/Health/allergy_information.json",
            f"user_{user_email}/ChatGPT/Health/Healthy_alternatives.json",
            f"user_{user_email}/ChatGPT/Health/healthy_eating_advice.json",
            f"user_{user_email}/ChatGPT/Health/Health_Advice.json",
            f"user_{user_email}/ChatGPT/Health/healthy_usage.json",
            f"user_{user_email}/ChatGPT/Health/Nutritional_Analysis.json",
            f"user_{user_email}/ChatGPT/Health/health_incompatibility_information_all.json",
            f"user_{user_email}/ChatGPT/Recipe/User_Defined_Dish.json",
            f"user_{user_email}/ChatGPT/Recipe/Fusion_Cuisine_Suggestions.json",
            f"user_{user_email}/ChatGPT/Recipe/Unique_Recipes.json",
            f"user_{user_email}/ChatGPT/Recipe/generated_recipes.json",
            f"user_{user_email}/ChatGPT/Recipe/diet_schedule.json",
        ]

        missing_files = [f for f in required_files if f not in existing_files]
        return (
            jsonify(
                {
                    "user_email": user_email,
                    "total_files": len(existing_files),
                    "categories": {
                        "ItemsList": {
                            "count": len(itemslist_files),
                            "files": itemslist_files,
                        },
                        "ChatGPT_HomePage": {
                            "count": len(chatgpt_homepage_files),
                            "files": chatgpt_homepage_files,
                        },
                        "ChatGPT_Health": {
                            "count": len(chatgpt_health_files),
                            "files": chatgpt_health_files,
                        },
                        "ChatGPT_Recipe": {
                            "count": len(chatgpt_recipe_files),
                            "files": chatgpt_recipe_files,
                        },
                        "User_Info": {
                            "count": len(user_info_files),
                            "files": user_info_files,
                        },
                        "Improper_Location": {
                            "count": len(improper_files),
                            "files": improper_files,
                        },
                    },
                    "missing_files": {
                        "count": len(missing_files),
                        "files": missing_files,
                    },
                    "required_files_total": len(required_files),
                    "setup_complete": len(missing_files) == 0
                    and len(improper_files) == 0,
                }
            ),
            200,
        )
    except Exception as exc:
        logging.error(f"Error checking user files: {exc}")
        return jsonify({"error": "An internal error occurred."}), 500


def cleanup_user_files_function():
    """Delete user files that are outside the expected folder structure."""
    try:
        user_email = core.get_user_email_from_token()
        logging.info(f"Cleaning up files for user: {user_email}")

        prefix = f"user_{user_email}/"
        blobs = core.bucket.list_blobs(prefix=prefix)
        allowed_patterns = [
            f"user_{user_email}/ItemsList/",
            f"user_{user_email}/ChatGPT/HomePage/",
            f"user_{user_email}/ChatGPT/Health/",
            f"user_{user_email}/ChatGPT/Recipe/",
            f"user_{user_email}/.user_info.json",
        ]

        files_to_delete = []
        files_to_keep = []
        for blob in blobs:
            file_path = blob.name
            if any(file_path.startswith(pattern) for pattern in allowed_patterns):
                files_to_keep.append(file_path)
            else:
                files_to_delete.append(file_path)

        deleted_files = []
        failed_deletions = []
        for file_path in files_to_delete:
            try:
                core.bucket.blob(file_path).delete()
                deleted_files.append(file_path)
            except Exception as exc:
                failed_deletions.append(
                    {"file": file_path, "error": "Internal deletion error"}
                )
                logging.error(f"Failed to delete {file_path}: {exc}")

        return (
            jsonify(
                {
                    "message": "File cleanup completed",
                    "user_email": user_email,
                    "summary": {
                        "files_kept": len(files_to_keep),
                        "files_deleted": len(deleted_files),
                        "failed_deletions": len(failed_deletions),
                    },
                    "details": {
                        "kept_files": files_to_keep,
                        "deleted_files": deleted_files,
                        "failed_deletions": failed_deletions,
                    },
                }
            ),
            200,
        )
    except Exception as exc:
        logging.error(f"Error cleaning up user files: {exc}")
        return jsonify({"error": "An internal error occurred."}), 500


def initialize_user_complete_function():
    """Create the default storage files for a newly initialized user."""
    try:
        user_email = core.get_user_email_from_token()
        logging.info(f"Complete initialization for user: {user_email}")

        required_files = {
            "ItemsList/master_nonexpired.json": {"Food": [], "Not_Food": []},
            "ItemsList/master_expired.json": {"Food": [], "Not_Food": []},
            "ItemsList/shopping_list.json": {"Food": [], "Not_Food": []},
            "ItemsList/result.json": {"Food": [], "Not_Food": []},
            "ItemsList/item_frequency.json": {"Food": []},
            "ChatGPT/HomePage/food_handling_advice.json": [],
            "ChatGPT/HomePage/Food_Waste_Reduction_Suggestions.json": [],
            "ChatGPT/HomePage/Ethical_Eating_Suggestions.json": [],
            "ChatGPT/HomePage/generated_fun_facts.json": [],
            "ChatGPT/HomePage/Cooking_Tips.json": [],
            "ChatGPT/HomePage/Current_Trends.json": [],
            "ChatGPT/HomePage/Mood_Changer.json": [],
            "ChatGPT/HomePage/Joke.json": {
                "last_generated": datetime.min.isoformat(),
                "jokes": [],
            },
            "ChatGPT/Health/generated_nutritional_advice.json": [],
            "ChatGPT/Health/allergy_information.json": [],
            "ChatGPT/Health/Healthy_alternatives.json": [],
            "ChatGPT/Health/healthy_eating_advice.json": [],
            "ChatGPT/Health/Health_Advice.json": [],
            "ChatGPT/Health/healthy_usage.json": [],
            "ChatGPT/Health/Nutritional_Analysis.json": [],
            "ChatGPT/Health/health_incompatibility_information_all.json": [],
            "ChatGPT/Recipe/User_Defined_Dish.json": [],
            "ChatGPT/Recipe/Fusion_Cuisine_Suggestions.json": [],
            "ChatGPT/Recipe/Unique_Recipes.json": [],
            "ChatGPT/Recipe/generated_recipes.json": [],
            "ChatGPT/Recipe/diet_schedule.json": [],
        }

        created_files = []
        existing_files = []
        failed_files = []

        for relative_path, default_data in required_files.items():
            try:
                full_path = f"user_{user_email}/{relative_path}"
                blob = core.bucket.blob(full_path)
                if blob.exists():
                    existing_files.append(full_path)
                else:
                    blob.upload_from_string(
                        json.dumps(default_data, indent=4), if_generation_match=0
                    )
                    test_blob = core.bucket.blob(full_path)
                    if not test_blob.exists():
                        failed_files.append(
                            {
                                "file": full_path,
                                "error": "Blob verification failed after upload",
                            }
                        )
                    else:
                        created_files.append(full_path)
            except Exception as exc:
                failed_files.append(
                    {"file": full_path, "error": "Internal creation error"}
                )
                logging.error(f"Failed to create {full_path}: {exc}")

        return (
            jsonify(
                {
                    "message": "User initialization completed",
                    "user_email": user_email,
                    "summary": {
                        "created": len(created_files),
                        "existing": len(existing_files),
                        "failed": len(failed_files),
                    },
                    "details": {
                        "created_files": created_files,
                        "existing_files": existing_files,
                        "failed_files": failed_files,
                    },
                }
            ),
            200,
        )
    except Exception as exc:
        logging.error(f"Error in complete initialization: {exc}")
        return jsonify({"error": "An internal error occurred."}), 500


def initialize_user_data_if_needed(user_email):
    """Ensure a user's master inventory file exists before processing uploads."""
    try:
        blob = core.bucket.blob(f"user_{user_email}/ItemsList/master_nonexpired.json")
        return blob.exists()
    except Exception as exc:
        logging.error(f"Error checking user data: {exc}")
        return False


def set_email_create_function():
    """Create Firestore and GCS storage entries for the authenticated user."""
    email = None
    uid = None
    try:
        auth_header = request.headers.get("Authorization")
        data = request.get_json(silent=True) or {}
    except Exception as exc:
        logging.error(f"Error in set_email_create_function: {exc}")
        data = {}

    id_token = None
    if auth_header and auth_header.startswith("Bearer "):
        id_token = auth_header.split("Bearer ")[1]
    else:
        id_token = data.get("idToken")

    if not id_token:
        return jsonify({"error": "No token provided"}), 400

    decoded_token = core.auth.verify_id_token(id_token, clock_skew_seconds=60)
    uid = decoded_token.get("uid")
    email = decoded_token.get("email")

    if not email:
        return jsonify({"error": "Email missing in token"}), 400

    safe_email = core.sanitize_email(email)

    try:
        db = firestore.client()
        user_ref = db.collection("users").document(uid)
        user_doc = user_ref.get()
        if not user_doc.exists:
            user_ref.set({"email": email, "created_at": firestore.SERVER_TIMESTAMP})
        else:
            logging.info(f"User {email} already exists in Firestore")
    except Exception as exc:
        logging.error(f"Firestore error: {exc}")
        return jsonify({"error": f"Error creating user record: {exc}"}), 500

    try:
        user_info_path = f"user_{safe_email}/.user_info.json"
        blob = core.bucket.blob(user_info_path)
        if not blob.exists():
            user_info = {
                "email": email,
                "uid": uid,
                "created_at": datetime.now().isoformat(),
                "folders": [
                    "ItemsList",
                    "ChatGPT/HomePage",
                    "ChatGPT/Health",
                    "ChatGPT/Recipe",
                ],
            }
            blob.upload_from_string(
                json.dumps(user_info, indent=4), content_type="application/json"
            )

        required_files = {
            f"user_{safe_email}/ItemsList/master_nonexpired.json": {
                "Food": [],
                "Not_Food": [],
            },
            f"user_{safe_email}/ItemsList/master_expired.json": {
                "Food": [],
                "Not_Food": [],
            },
            f"user_{safe_email}/ItemsList/shopping_list.json": {
                "Food": [],
                "Not_Food": [],
            },
            f"user_{safe_email}/ItemsList/result.json": {"Food": [], "Not_Food": []},
            f"user_{safe_email}/ItemsList/item_frequency.json": {"Food": []},
            f"user_{safe_email}/ChatGPT/HomePage/food_handling_advice.json": [],
            f"user_{safe_email}/ChatGPT/HomePage/Food_Waste_Reduction_Suggestions.json": [],
            f"user_{safe_email}/ChatGPT/HomePage/Ethical_Eating_Suggestions.json": [],
            f"user_{safe_email}/ChatGPT/HomePage/generated_fun_facts.json": [],
            f"user_{safe_email}/ChatGPT/HomePage/Cooking_Tips.json": [],
            f"user_{safe_email}/ChatGPT/HomePage/Current_Trends.json": [],
            f"user_{safe_email}/ChatGPT/HomePage/Mood_Changer.json": [],
            f"user_{safe_email}/ChatGPT/HomePage/Joke.json": {
                "last_generated": datetime.min.isoformat(),
                "jokes": [],
            },
            f"user_{safe_email}/ChatGPT/Health/generated_nutritional_advice.json": [],
            f"user_{safe_email}/ChatGPT/Health/allergy_information.json": [],
            f"user_{safe_email}/ChatGPT/Health/Healthy_alternatives.json": [],
            f"user_{safe_email}/ChatGPT/Health/healthy_eating_advice.json": [],
            f"user_{safe_email}/ChatGPT/Health/Health_Advice.json": [],
            f"user_{safe_email}/ChatGPT/Health/healthy_usage.json": [],
            f"user_{safe_email}/ChatGPT/Health/Nutritional_Analysis.json": [],
            f"user_{safe_email}/ChatGPT/Health/health_incompatibility_information_all.json": [],
            f"user_{safe_email}/ChatGPT/Recipe/User_Defined_Dish.json": [],
            f"user_{safe_email}/ChatGPT/Recipe/Fusion_Cuisine_Suggestions.json": [],
            f"user_{safe_email}/ChatGPT/Recipe/Unique_Recipes.json": [],
            f"user_{safe_email}/ChatGPT/Recipe/generated_recipes.json": [],
            f"user_{safe_email}/ChatGPT/Recipe/diet_schedule.json": [],
        }

        created_files = []
        skipped_files = []
        for file_path, default_data in required_files.items():
            blob = core.bucket.blob(file_path)
            if blob.exists():
                skipped_files.append(file_path)
            else:
                try:
                    blob.upload_from_string(
                        json.dumps(default_data, indent=4), if_generation_match=0, content_type="application/json"
                    )
                    created_files.append(file_path)
                except Exception as exc:
                    logging.warning(f"Skipping {file_path} due to creation error: {exc}")
                    skipped_files.append(file_path)

        return (
            jsonify(
                {
                    "message": "User setup completed successfully",
                    "email": email,
                    "files_created": len(created_files),
                    "files_skipped": len(skipped_files),
                }
            ),
            200,
        )
    except Exception as exc:
        logging.error(f"Error setting up user storage: {exc}")
        return jsonify({"error": "An internal error occurred."}), 500


def update_purchased_nonexpired_shopping_item_price_function():
    """Update the stored price for an item in the inventory collections."""
    try:
        data = request.get_json(force=True)
        item_name = data["Name"].lower()
        new_price = float(data["Price"])

        master_data = get_data_from_json("ItemsList", "master_nonexpired")
        shopping_list_data = get_data_from_json("ItemsList", "shopping_list")
        result_data = get_data_from_json("ItemsList", "result")

        def update_price(data_payload, item_name, new_price):
            item_found = False
            for category, items in data_payload.items():
                for item in items:
                    item_name_in_data = item.get("Name", item.get("name", "")).lower()
                    if item_name_in_data == item_name.lower():
                        item["Price"] = f"${new_price:.2f}"
                        item_found = True
            return item_found

        item_found_in_master = update_price(master_data, item_name, new_price)
        item_found_in_shopping_list = update_price(
            shopping_list_data, item_name, new_price
        )
        item_found_in_result = update_price(result_data, item_name, new_price)

        if (
            not item_found_in_master
            and not item_found_in_shopping_list
            and not item_found_in_result
        ):
            return jsonify({"message": "Item not found in any list."}), 404

        save_data_to_cloud_storage("ItemsList", "master_nonexpired", master_data)
        save_data_to_cloud_storage("ItemsList", "shopping_list", shopping_list_data)
        save_data_to_cloud_storage("ItemsList", "result", result_data)

        updated_lists = []
        if item_found_in_master:
            updated_lists.append("master_nonexpired")
        if item_found_in_shopping_list:
            updated_lists.append("shopping_list")
        if item_found_in_result:
            updated_lists.append("purchased_items")

        return (
            jsonify(
                {
                    "message": "Price updated successfully",
                    "updated_in": updated_lists,
                    "item_name": item_name,
                    "new_price": f"${new_price:.2f}",
                }
            ),
            200,
        )
    except ValueError:
        return jsonify({"error": "Invalid data provided."}), 400
    except Exception as exc:
        return jsonify({"error": "An internal error occurred."}), 500


def get_data_from_json(folder_name, file_name):
    """Compatibility wrapper for the app-level helper."""
    return core.get_data_from_json(folder_name, file_name)


def save_data_to_cloud_storage(folder_name, file_name, data, max_retries=5):
    """Compatibility wrapper for the app-level helper."""
    return core.save_data_to_cloud_storage(
        folder_name, file_name, data, max_retries=max_retries
    )
