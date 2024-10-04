from datetime import date, datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

import cv2
import json
import os
import re
import tempfile
import base64
import random
import time
import logging
from functools import wraps
import chardet

import calendar

from PIL import Image
import pytesseract

from flask import Flask, jsonify, request
from flask_cors import CORS

from dateparser.search import search_dates

from google.cloud import secretmanager_v1, storage
from google.oauth2 import service_account
from google.api_core.exceptions import DeadlineExceeded
from google.api_core.exceptions import NotFound
from google.resumable_media.common import InvalidResponse

import firebase_admin
from firebase_admin import credentials, firestore, auth

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "https://my-grocery-home.uc.r.appspot.com"}})

language = "eng"
clock_skew_seconds = 60
text = ""
date_record = list()
# Setting Environment Variables 
os.environ["BUCKET_NAME"] = "my-grocery"
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-grocery-home"
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
bucket_name = os.environ.get("BUCKET_NAME")

# logging.basicConfig(level=logging.DEBUG)
# Create a Secret Manager client and Access Service Account Key
client = secretmanager_v1.SecretManagerServiceClient()

                                            # Credentials Section
##############################################################################################################################################################################
# Function to access the Secret Manager
def access_secret_version(client, project_id, secret_id, timeout=60):
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name}, timeout=timeout)
    payload = response.payload.data.decode("UTF-8")
    return payload
# --------------------------------------------------------------------------------------------------------

# Function to initialize Firebase
def initialize_firebase():
    firebase_secret_id = 'firebase_service_account'
    retries = 5
    for attempt in range(retries):
        try:
            firebase_cred_data = access_secret_version(client, project_id, firebase_secret_id)
            firebase_cred_dict = json.loads(firebase_cred_data)
            cred = credentials.Certificate(firebase_cred_dict)
            firebase_admin.initialize_app(cred)
            global db
            db = firestore.client()
            print("Firebase credentials retrieved and app initialized successfully.")
            break
        except DeadlineExceeded:
            print(f"Attempt {attempt + 1} failed: Deadline Exceeded. Retrying...")
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(sleep_time)  # Exponential backoff with jitter
        except Exception as e:
            print("Error initializing Firebase app:", e)
            break
initialize_firebase()
# --------------------------------------------------------------------------------------------------------

# Initialize OpenAi API and Google cloud credentials
try:
    service_account_secret_id = 'my-credentials-json'
    service_account_key = access_secret_version(client, project_id, service_account_secret_id)
    credentials = service_account.Credentials.from_service_account_info(json.loads(service_account_key))
    storage_client = storage.Client(credentials=credentials, project=project_id)
    bucket = storage_client.bucket(bucket_name)
    print("Service Account Key retrieved successfully.")
except Exception as e:
    print("Error retrieving service account key:", e)

try:
    app_default_secret_id = 'Application-default-credentials'
    app_default_credentials = access_secret_version(client, project_id, app_default_secret_id).strip()
    print("Application Default Credentials retrieved successfully.")
except Exception as e:
    print("Error retrieving application default credentials:", e)

try:
    openai_secret_id = 'OPENAI-API-KEY'
    openai_api_key = access_secret_version(client, project_id, openai_secret_id)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    import openai
    openai.api_key = openai_api_key  
    print("OpenAI API key retrieved successfully.")
except Exception as e:
    print("Error retrieving OpenAI API key from Google Secret Manager:", e)

                                                # Main code functions
##############################################################################################################################################################################

# Function to authenticate user
def authenticate_user_function(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        id_token = request.headers.get('Authorization')
        if not id_token:
            return jsonify({'error': 'No token provided'}), 401
        try:
            # Remove 'Bearer ' from token
            id_token = id_token.split(' ')[1]
            decoded_token = auth.verify_id_token(id_token)
            # You can add the user to the request object if needed
            # request.user = decoded_token
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
    return decorated_function
# --------------------------------------------------------------------------------------------------------

# Funcion to set user email in firestore
def get_user_email_from_token():
    try:
        id_token = request.headers.get('Authorization')
        if id_token:
            id_token = id_token.split('Bearer ')[1]
            decoded_token = auth.verify_id_token(id_token, clock_skew_seconds=clock_skew_seconds)
            return decoded_token['email']
        else:
            raise Exception("Authorization header missing")
    except Exception as e:
        raise Exception(f"Failed to get user email from token: {str(e)}")
# --------------------------------------------------------------------------------------------------------

# Function to add items to list       
def add_item_to_list(master_list_name, slave_list_name):
    try:
        item_name = request.json.get("itemName").lower()
        if not item_name:
            return jsonify({"error": "Item name is missing in the request body"}), 400
        master_data = get_data_from_json("ItemsList", master_list_name)
        if not isinstance(master_data, dict):
            return jsonify({"error": f"Invalid data format in {master_list_name}.json"}), 500
        # Find the item in master list
        for category, items in master_data.items():
            for item in items:
                if item["Name"].lower() == item_name:
                    slave_data = get_data_from_json("ItemsList", slave_list_name)
                    if category in slave_data:
                        slave_data[category].append(item)
                    else:
                        slave_data[category] = [item]
                    response = {
                        "Food": slave_data.get("Food", []),
                        "Not_Food": slave_data.get("Not_Food", []),
                    }
                    save_data_to_cloud_storage("ItemsList", slave_list_name, response)
                    return jsonify({"message": f"Item '{item_name}' added to {slave_list_name} successfully"}), 200
        return jsonify({"error": f"Item '{item_name}' not found in {master_list_name}.json"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# --------------------------------------------------------------------------------------------------------

# Function to move item from food to non food
def move_to_food_function():
    try:
        item_name = request.json.get('itemName')
        if not item_name:
            return jsonify({"error": "Item name is missing"}), 400
        # Fetch both datasets
        master_data = get_data_from_json("ItemsList", "master_nonexpired")
        if "error" in master_data:
            return jsonify({"error": f"Failed to retrieve master data: {master_data['error']}"}), 500
        expired_data = get_data_from_json("ItemsList", "master_expired")
        print(f"Expired data type: {type(expired_data)}, content: {expired_data}")
        if "error" in expired_data:
            return jsonify({"error": f"Failed to retrieve expired data: {expired_data['error']}"}), 500
        result_data = get_data_from_json("ItemsList", "result")
        if "error" in result_data:
            return jsonify({"error": f"Failed to retrieve result data: {result_data['error']}"}), 500
        # Process master_nonexpired data
        for item in master_data['Not_Food']:
            if item['Name'] == item_name:
                master_data['Food'].append(item)
                master_data['Not_Food'].remove(item)
                break
        # Process expired_data
        # Check if expired_data is a dictionary
        if not isinstance(expired_data, dict):
            return jsonify({"error": "Invalid data format for expired_data, expected a dictionary"}), 500
        # Process expired_data
        for category, items in expired_data.items():
            if not isinstance(items, list):
                return jsonify({"error": f"Invalid data format in expired_data for category: {category}"}), 500
            for item in items:
                if item['Name'] == item_name and category != 'Food':
                    master_data['Food'].append(item)
                    expired_data[category].remove(item)
                    break
        # Save updated datasets
        save_data_to_cloud_storage("ItemsList", "master_nonexpired", master_data)
        save_data_to_cloud_storage("ItemsList", "expired_data", expired_data)
        # Update result.json
        # Find the moved item in master_data or expired_data
        moved_item = None
        for item in master_data['Food']:
            if item['Name'] == item_name:
                moved_item = item
                break
        if moved_item:
            # Add the moved item to the Food category in result.json
            if 'Food' not in result_data:
                result_data['Food'] = []
            result_data['Food'].append(moved_item) 
            # Remove the item from its original category in result.json if present
            for category in result_data:
                if category != 'Food':
                    result_data[category] = [item for item in result_data[category] if item['Name'] != item_name]
            # Save updated result.json
            save_data_to_cloud_storage("ItemsList", "result", result_data)
        return jsonify({"message": "Item moved to Food successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# --------------------------------------------------------------------------------------------------------

# Function to change item name
def update_item_name():
    try:
        data = request.json
        old_name = data['oldName']
        new_name = data['newName']
        category = data['category']
        # Load the current data
        nonexpired_content = get_data_from_json("ItemsList", "master_nonexpired")
        result_content = get_data_from_json("ItemsList", "result")
        
        if isinstance(nonexpired_content, bytes):
            nonexpired_content = nonexpired_content.decode('utf-8')
        if isinstance(nonexpired_content, str):
            nonexpired_content = json.loads(nonexpired_content)
        # Update the item name
        for item in nonexpired_content['Food', 'Not_Food']:
            if item['Name'] == old_name:
                item['Name'] = new_name
                break
        save_data_to_cloud_storage("ItemsList", "master_nonexpired", nonexpired_content)
        
        if isinstance(result_content, bytes):
            result_content = result_content.decode('utf-8')
        if isinstance(result_content, str):
            result_content = json.loads(result_content)
        # Update the item name
        for item in result_content['Food', 'Not_Food']:
            if item['Name'] == old_name:
                item['Name'] = new_name
                break
        # Save the updated data
        save_data_to_cloud_storage("ItemsList", "result", result_content)
        return jsonify({"message": "Item name updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# --------------------------------------------------------------------------------------------------------

# Function to add custom item to list
def add_custom_item_function():
    data = get_data_from_json("ItemsList", "shopping_list")
    # Get user input for name and category
    request_data = request.get_json()
    item_name = request_data.get("item_name").lower()
    item_price = request_data.get("item_price")
    item_date = request_data.get("item_date")
    category = "Food"
    # Define default item details
    default_item = {
        "Name": "TestFNE",
        "Price": "$0.0",
        "Date": "8/1/2016",
        "Image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTvV8GjIu4AF9h-FApH1f1mkzktVXY7lhI5SDqd60AeKZtMSE6Nlpmvw7aO_Q&s",
    }
    # Create a new item dictionary
    new_item = default_item.copy()
    new_item["Name"] = item_name.lower()
    new_item["Price"] = item_price
    new_item["Date"] = item_date
    # Add the new item to the respective category
    if category == "Food":
        data["Food"].append(new_item)
    elif category == "Not_Food":
        data["Not_Food"].append(new_item)
    else:
        print("Invalid category. Please choose 'Food' or 'Not Food'.")
    # Save the updated data
    response = {
        "Food": data["Food"],
        "Not_Food": data["Not_Food"],
    }
    save_data_to_cloud_storage( "ItemsList", "shopping_list", response)
    return jsonify({"message": "Expiry updated successfully"})
# --------------------------------------------------------------------------------------------------------

# Function to delete all items from list
def delete_all_items(file_type):
    data = get_data_from_json("ItemsList", file_type)
    # Filter the items
    data["Food"] = [item for item in data["Food"] if item["Name"] == "TestFNE"]
    data["Not_Food"] = [item for item in data["Not_Food"] if item["Name"] == "TestFNE"]
    # Save the updated data
    response = {
        "Food": data["Food"],
        "Not_Food": data["Not_Food"],
    }
    save_data_to_cloud_storage("ItemsList", file_type, response)
    return jsonify({"message": f"{file_type.replace('_', ' ')} list deleted successfully"})
# --------------------------------------------------------------------------------------------------------

# Function to delete individual items from list
def delete_item_from_list(list_name):
    try:
        json_data = get_data_from_json("ItemsList", list_name)
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
            save_data_to_cloud_storage("ItemsList", list_name, response)
            return jsonify({"message": f"Item '{item_name}' deleted successfully"}), 200
        else:
            return jsonify({"message": f"Item '{item_name}' not found in the JSON data"}), 404
    except Exception as e:
        return jsonify({"message": f"An error occurred while processing the request: {e}"}), 500
# --------------------------------------------------------------------------------------------------------

# Function to get file respose in base64 from cloud storage  
def get_file_response_base64(file_name):
    user_email = get_user_email_from_token()
    folder_name = f"user_{user_email}/ItemsList"
    json_blob_name = f"{folder_name}/{file_name}.json"
    json_blob = bucket.blob(json_blob_name)
    if json_blob.exists():  # Check if the blob exists
        data = json_blob.download_as_bytes()
        data_base64 = base64.b64encode(data).decode("utf-8")  # Encode as base64
        return jsonify({"data": data_base64})
    else:
        return jsonify({"message": "No JSON file found."}), 404
# --------------------------------------------------------------------------------------------------------

# Function to get data from google cloud storage    
def get_data_from_json(folder_name, file_name):
    """Downloads data from a storage bucket and returns it as JSON response."""
    user_email = get_user_email_from_token()# Replace with dynamic user email retrieval if needed
    json_blob_name = f"user_{user_email}/{folder_name}/{file_name}.json"
    blob = bucket.blob(json_blob_name)
    try:
        # Check if the file exists before attempting to download
        if blob.exists():
            content = blob.download_as_text()
            data = json.loads(content)
            return data
        else:
            return {"error": "File not found"}, 404
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

# --------------------------------------------------------------------------------------------------------

# Function to save data to google cloud storage
def save_data_to_cloud_storage(folder_name, file_name, data, max_retries=5):
    """Saves data to a JSON file in a storage bucket with exponential backoff."""
    user_email = get_user_email_from_token()  # Replace with dynamic user email retrieval if needed
    json_blob_name = f"user_{user_email}/{folder_name}/{file_name}.json"
    blob = bucket.blob(json_blob_name)
    attempt = 0
    backoff = 1  # Initial backoff time in seconds
    while attempt < max_retries:
        try:
            json_data = json.dumps(data, indent=4)
            blob.upload_from_string(json_data, content_type="application/json")
            logging.info(f"Data saved to {json_blob_name}")
            return {"message": "Data saved successfully"}, 200
        except InvalidResponse as e:
            if e.response.status_code == 429:  # Rate limit error
                logging.warning(f"Rate limit exceeded. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2  # Exponentially increase the backoff time
                attempt += 1
            else:
                logging.exception("Exception occurred while saving data to cloud storage")
                return {"error": str(e)}, 500
        except Exception as e:
            logging.exception("Unexpected exception occurred while saving data to cloud storage")
            return {"error": str(e)}, 500
    logging.error("Max retries exceeded. Failed to save the file.")
    return {"error": "Max retries exceeded. Failed to save the file."}, 500
# --------------------------------------------------------------------------------------------------------

# Function to read a JSON file and return its contents
def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data
# --------------------------------------------------------------------------------------------------------

# Function to calculate days left until expiry
def calculate_days_until_expiry(item):
    expiry_date = datetime.strptime(item["Expiry_Date"], "%d/%m/%Y")
    current_date = datetime.strptime(item["Date"], "%d/%m/%Y")
    days_until_expiry = (expiry_date - current_date).days
    return days_until_expiry
# --------------------------------------------------------------------------------------------------------

# Function to remove duplicates from master_nonexpired_data
def remove_duplicates_nonexpired(master_nonexpired_data):
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
                    existing_price_str = existing_price.replace("$", "").replace(",", "")
                else:
                    existing_price_str = str(existing_price)
                existing_price = float(existing_price_str)
                
                if price < existing_price:
                    unique_items[name] = item
        master_nonexpired_data[category] = list(unique_items.values())
        print("Successfully removed duplicates from nonexpired data")
    return master_nonexpired_data

# --------------------------------------------------------------------------------------------------------

# Function to remove duplicates from result
def remove_duplicates_result(result):
    for category, items in result.items():
        unique_items = {}
        for item in items:
            name = item["Name"].lower()  # Convert to lowercase for case-insensitive comparison
            price = item["Price"]
            if isinstance(price, str):
                price_str = price.replace("$", "").replace(",", "")
            else:
                price_str = str(price)
            price = float(price_str)
            if name not in unique_items or price < unique_items[name]['price']:
                unique_items[name] = {
                    'item': item,
                    'price': price
                }
        result[category] = [item_data['item'] for item_data in unique_items.values()]
        print("Successfully removed duplicates from result")
    return result
# --------------------------------------------------------------------------------------------------------

# Function to remove duplicates from data_expired
def remove_duplicates_expired(data_expired):
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
                    existing_price_str = existing_price.replace("$", "").replace(",", "")
                else:
                    existing_price_str = str(existing_price)
                existing_price = float(existing_price_str)
                if price < existing_price:
                    unique_items[name] = item
        data_expired[category] = list(unique_items.values())
        print("Successfully removed duplicates from expired data")
    return data_expired
# --------------------------------------------------------------------------------------------------------

# Function to remove items present in expired data from master_nonexpired_data
def remove_items_present_in_expired_from_nonexpired(master_nonexpired_data, master_expired_data):
    # Iterate through each category in the expired data
    for category, expired_items in master_expired_data.items():
        # For each expired item, check if it exists in the non-expired list
        nonexpired_items = master_nonexpired_data.get(category, [])
        # Create a list to store items to remove from non-expired data
        items_to_remove = []    
        for expired_item in expired_items:
            for nonexpired_item in nonexpired_items:
                # Compare items by "Name" and "Expiry_Date" to identify duplicates
                if (expired_item["Name"] == nonexpired_item["Name"] and
                    expired_item["Expiry_Date"] == nonexpired_item["Expiry_Date"]):
                    items_to_remove.append(nonexpired_item)
                    break  # No need to check further once a match is found
        # Remove the identified items from the non-expired list
        master_nonexpired_data[category] = [item for item in nonexpired_items if item not in items_to_remove]
    return master_nonexpired_data
# --------------------------------------------------------------------------------------------------------

# Function to update user enetered price in master_nonexpired_data
def update_nonexpired_shopping_list_item_price_function():
    try:
        # Parse incoming JSON data
        data = request.get_json(force=True)
        item_name = data["Name"].lower()
        new_price = float(data["Price"])  # Convert to float for price
        # Step 1: Retrieve and parse the master data and shopping list data from JSON files
        master_data = get_data_from_json("ItemsList", "master_nonexpired")
        shopping_list_data = get_data_from_json("ItemsList", "shopping_list")
        item_found_in_master = False  # Flag to track if item is found in master_nonexpired
        item_found_in_shopping_list = False  # Flag to track if item is found in shopping_list
        # Function to update price in the given dataset
        def update_price(data, item_name, new_price):
            item_found = False
            for category, items in data.items():
                for item in items:
                    if item["Name"].lower() == item_name:
                        item["Price"] = new_price  # Update the price
                        item_found = True
                        break  # Remove this if you want to update all instances of the item
            return item_found
        # Step 3: Find and update the price for the matching item in master_nonexpired
        item_found_in_master = update_price(master_data, item_name, new_price)
        # Step 4: Find and update the price for the matching item in shopping_list
        item_found_in_shopping_list = update_price(shopping_list_data, item_name, new_price)
        # If the item is not found in either master_nonexpired or shopping_list
        if not item_found_in_master and not item_found_in_shopping_list:
            return jsonify({"message": "Item not found."}), 404
        # Step 5: Save the updated data back to the storage
        save_data_to_cloud_storage("ItemsList", "master_nonexpired", master_data)
        save_data_to_cloud_storage("ItemsList", "shopping_list", shopping_list_data)
        # Return success response
        return jsonify({"message": "Price updated successfully"}), 200
    except ValueError as e:
        return jsonify({"error": "Invalid data provided."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# --------------------------------------------------------------------------------------------------------

# Function to append unique data from a JSON file to the master_nonexpired JSON data
def append_unique_to_master_nonexpired(master_nonexpired_data, data_to_append, category):
    for item_to_append in data_to_append[category]:
        item_unique = True
        for master_nonexpired_item in master_nonexpired_data[category]:
            # Check if all properties match
            if all(
                item_to_append[prop] == master_nonexpired_item[prop]
                for prop in ["Name", "Price", "Date", "Expiry_Date", "Status"]
            ):
                item_unique = False
                break
        if item_unique:
            # ---------------------------------------------
            days_until_expiry = calculate_days_until_expiry(item_to_append)
            item_to_append["Days_Until_Expiry"] = days_until_expiry
            # ---------------------------------------------
            master_nonexpired_data[category].append(item_to_append)      
    return master_nonexpired_data
# --------------------------------------------------------------------------------------------------------

# Function to process JSON files in a folder
def process_json_files_folder(temp_dir):
    master_nonexpired_data = get_data_from_json("ItemsList", "master_nonexpired")
    json_files_to_append = [f for f in os.listdir(temp_dir) if f.endswith(".json")]
    # JSON file path in the temp_dir
    for json_file in json_files_to_append:
        json_file_path = os.path.join(temp_dir, json_file)
        data_to_append = read_json_file(json_file_path)
    if os.path.isfile(json_file_path):
        # Append data to both "Food" and "Not Food" categories
        append_unique_to_master_nonexpired(master_nonexpired_data, data_to_append, "Food")
        append_unique_to_master_nonexpired(master_nonexpired_data, data_to_append, "Not_Food")
    else:
        print(f"JSON file not found at {json_file_path}")
    # ------------------------------------------------------------------------------------------------------
    # Write the updated master_nonexpired JSON data back to the file
    save_data_to_cloud_storage("ItemsList", "master_nonexpired", master_nonexpired_data )
    return jsonify({"message": "Data appended successfully"})
# --------------------------------------------------------------------------------------------------------

# Function to update the expiry date of a specific item in both master_nonexpired and shopping_list
def update_master_nonexpired_item_expiry_function():
    try:
        # Parse incoming JSON data
        data = request.get_json(force=True)
        item_name = data["item_name"].lower()
        days_to_extend = int(data["days_to_extend"])  # Convert to integer
        # Step 1: Retrieve and parse the master data from JSON file
        master_data = get_data_from_json("ItemsList", "master_nonexpired") 
        shopping_data = get_data_from_json("ItemsList", "shopping_list")   
        # Step 2: Find and update the expiry date in master data
        item_found = False  # To track if the item is found
        for category, items in master_data.items():
            for item in items:
                if item["Name"].lower() == item_name:
                    expiry_date = datetime.strptime(item["Expiry_Date"], "%d/%m/%Y")
                    new_expiry_date = expiry_date + timedelta(days=days_to_extend)
                    item["Expiry_Date"] = new_expiry_date.strftime("%d/%m/%Y")
                    item['Days_Until_Expiry'] += days_to_extend 
                    item["Status"] = "Not Expired"
                    item_found = True
                    break  # Exit the loop once item is found and updated
        # Step 3: Find and update the expiry date in shopping list data
        for category, items in shopping_data.items():
            for item in items:
                if item["Name"].lower() == item_name:
                    expiry_date = datetime.strptime(item["Expiry_Date"], "%d/%m/%Y")
                    new_expiry_date = expiry_date + timedelta(days=days_to_extend)
                    item["Expiry_Date"] = new_expiry_date.strftime("%d/%m/%Y")
                    item['Days_Until_Expiry'] += days_to_extend 
                    item["Status"] = "Not Expired"
                    break  # Exit the loop once item is found and updated
        # If item not found in master data
        if not item_found:
            return jsonify({"message": "Item not found in master_nonexpired."}), 404
        # Step 4: Save updated data back to cloud storage
        save_data_to_cloud_storage("ItemsList", "master_nonexpired", master_data)
        save_data_to_cloud_storage("ItemsList", "shopping_list", shopping_data)
        # Call the function to update the expiry database (assumed to update another database or perform further operations)
        update_expiry_database_user_defined(days_to_extend, item_name) 
        # Return a success response as JSON
        return jsonify({"message": "Expiry of the item updated successfully in both master_nonexpired and shopping_list."})
    except ValueError:
        return jsonify({"error": "Invalid input data provided."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# --------------------------------------------------------------------------------------------------------

# Function to update expiry database
def update_expiry_database_user_defined(days_to_extend, item_name):
    data = request.get_json(force=True)
    item_name = data["item_name"].lower()  # Convert item name to lowercase
    days_to_extend = data["days_to_extend"]
    # Step 1: Read and Process the Text File
    with open("items_expiry.txt", "r") as file:
        lines = file.readlines()
    products = [line.strip().split(",") for line in lines]
    # Step 2: Find and Update the Days to Expire (Enforce Lowercase)
    for i, (name, days) in enumerate(products):
        if name.lower() == item_name:  # Convert product name to lowercase for comparison
            products[i] = (name, str(days_to_extend))
            break
    # Step 3: Write Updated Data Back to Text File
    with open("items_expiry.txt", "w") as file:
        for name, days in products:
            file.write(f"{name},{days}\n")
    return jsonify({"message": "Expiry database updated successfully"})
# --------------------------------------------------------------------------------------------------------

# Function to create a master_expired file
def create_master_expired_file(data_nonexpired):
    # Load the existing expired items data
    try:
        data_expired = get_data_from_json("ItemsList", "master_expired")
    except FileNotFoundError:
        data_expired = {"Food": [], "Not_Food": []}
    # Get today's date in the correct format
    today = datetime.today().strftime("%d/%m/%Y")
    # Create a list to store items that should be removed from master_nonexpired JSON
    items_to_remove = []
    # Iterate through all items in nonexpired data to check for expired items
    for category, items in data_nonexpired.items():
        for item in items:
            expiry_date_str = item["Expiry_Date"]
            expiry_date = datetime.strptime(expiry_date_str, "%d/%m/%Y")
            today_date = datetime.strptime(today, "%d/%m/%Y")
            # Check if the item has expired
            if expiry_date < today_date:
                # Check if the item is already in the expired list to avoid duplication
                if not any(
                    expired_item.get("Name") == item["Name"] and
                    expired_item.get("Expiry_Date") == expiry_date_str
                    for expired_item in data_expired[category]
                ):
                    # Move item to the expired list and update its status
                    item["Status"] = "Expired"
                    data_expired[category].append(item)
                    # Add the item to the removal list
                    items_to_remove.append(item)
    # Remove the expired items from the master_nonexpired data
    for category in data_nonexpired:
        data_nonexpired[category] = [item for item in data_nonexpired[category] if item not in items_to_remove]
    # Ensure no duplicates remain in both expired and non-expired lists
    data_nonexpired = remove_items_present_in_expired_from_nonexpired(data_nonexpired, data_expired)
    # Remove duplicates from the expired data
    remove_duplicates_expired(data_expired)
    remove_duplicates_nonexpired(data_nonexpired)
    # Write the updated data back to the cloud storage
    save_data_to_cloud_storage("ItemsList", "master_expired", data_expired)
    save_data_to_cloud_storage("ItemsList", "master_nonexpired", data_nonexpired)
    print("Master Expired and Nonexpired files created and filtered successfully.")
    return jsonify({"message": "Expired item list created successfully"})
# --------------------------------------------------------------------------------------------------------

# Function to clean and sort files
def clean_and_sort_files(filenames):
    for filename in filenames:
        items = {}     
        # Detect file encoding
        with open(filename, 'rb') as raw_file:
            raw_data = raw_file.read()
            detected = chardet.detect(raw_data)
            encoding = detected['encoding']  
        # Read file with detected encoding
        with open(filename, "r", encoding=encoding, errors='ignore') as file:
            for line in file:
                parts = line.strip().lower().split(',')
                name = parts[0].strip()
                days = parts[-1].strip() if len(parts) > 1 else ''
                if name not in items or (days.isdigit() and int(days) < items[name]):
                    items[name] = int(days) if days.isdigit() else ''
        # Write sorted items back to file
        with open(filename, "w", encoding='utf-8') as file:
            for name, days in sorted(items.items()):
                file.write(f"{name},{days}\n" if days != '' else f"{name}\n")
    print("All lists have been cleaned and sorted successfully.")
# Usage remains the same
filenames = [
    "items_expiry.txt",
    "NonFoodItems.txt",
    "Kitchen_Eatables_Database.txt",
    "Irrelevant.txt",
    "ItemCost.txt"
]
clean_and_sort_files(filenames)
# --------------------------------------------------------------------------------------------------------

# Function to set mail and create database on GCS
def set_email_create_function():
    data = request.get_json()
    id_token = data['idToken']
    clock_skew_seconds = 60  # 60 seconds clock skew allowance
    try:
        decoded_token = auth.verify_id_token(id_token, clock_skew_seconds=clock_skew_seconds)
        uid = decoded_token['uid']
        email = decoded_token['email']
        # Log the current time and the token's issued-at time in both epoch and human-readable formats
        current_time = int(time.time())
        current_time_readable = datetime.fromtimestamp(current_time).isoformat()
        token_iat_readable = datetime.fromtimestamp(decoded_token['iat']).isoformat()
        print(f"Current time: {current_time} ({current_time_readable})")
        print(f"Token issued-at time: {decoded_token['iat']} ({token_iat_readable})")
        # Retrieve email directly from decoded token
        # Retrieve user data from Firestore
        db = firestore.client()
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()
        if not user_doc.exists:
            # Store user email in Firestore if not already stored
            user_ref.set({'email': email})
        # Create a folder for the user using the email address in Google Cloud Storage
        folder_name = f"user_{email}/"
        blob = bucket.blob(folder_name)  # Creating a file as a placeholder
        blob.upload_from_string('')  # Upload an empty string to create the folder
        
        local_data_folder = './Data-Folder'  # Replace with your local data folder path
        for root, dirs, files in os.walk(local_data_folder):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_data_folder)
                destination_blob = bucket.blob(f"{folder_name}{relative_path}")
                destination_blob.upload_from_filename(local_file_path)
        return jsonify({'message': 'User email, folder, and data files created and uploaded successfully', 'email': email}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# --------------------------------------------------------------------------------------------------------
    
# Function for OCR comparison with captured image
def compare_image_function():
    try:
        if "file" not in request.files:
            return jsonify({"message": "No file provided"}), 400
        file = request.files["file"]
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = file.filename
            file_path = os.path.join(temp_dir, filename)
            # Save the uploaded file to the temporary directory
            file.save(file_path)
            if filename != "dummy.jpg":
                ocr_text = ""
                raw_text = process_image(file_path) 
                ocr_lines = raw_text.split("\n")  # Split the text by lines
                cleaned_lines = [line.strip() for line in ocr_lines if line.strip()]  # Clean up each line
                # Join the lines back together with appropriate line breaks
                ocr_text = "\n".join(cleaned_lines)  # Extract text from the image       
            return jsonify({"message": "Image Processed successfully", "ocrText": ocr_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500    
# --------------------------------------------------------------------------------------------------------

# Function to perform main function of the code 
def main_function():
    try:
        blob = None  # Initialize blob with a default value
        result = None  # Initialize result with a default value
        if "file" not in request.files:
            return jsonify({"message": "No file provided"}), 400
        file = request.files["file"]
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = file.filename
            file_path = os.path.join(temp_dir, filename)
            # Upload file to Google Cloud Storage
            blob = storage_client.bucket(bucket_name).blob(filename)
            file.save(file_path)
            blob.upload_from_filename(file_path)
            kitchen_items = read_kitchen_eatables()
            nonfood_items = nonfood_items_list()
            irrelevant_names = irrelevant_names_list()
            # Process uploaded file (example: text extraction and processing)
            if filename != "dummy.jpg":
                text = process_image(file_path)
                result = process_text(text, kitchen_items, nonfood_items, irrelevant_names)
                if result is not None:  # Check if result was processed
                    remove_duplicates_result(result)
                    save_data_to_cloud_storage("ItemsList", "result", result)                   
                temp_file_path = os.path.join(temp_dir, "temp_data.json")
                with open(temp_file_path, "w") as json_file:
                    json.dump(result, json_file, indent=4)
                process_json_files_folder(temp_dir)
            # Example operations with master files
            data_nonexpired = get_data_from_json("ItemsList", "master_nonexpired")
            create_master_expired_file(data_nonexpired)
            # Upload processed data to storage
            save_data_to_cloud_storage("ItemsList", "master_nonexpired", data_nonexpired)
            try:
                # Attempt to delete the file if it exists
                blob.reload()  # Ensure the file still exists before deleting
                blob.delete()
            except NotFound:
                print("File not found during deletion, skipping.")
        return jsonify({"message": "File uploaded and processed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# --------------------------------------------------------------------------------------------------------

# Function to process image files
def process_image(file_path):
    try:
        with Image.open(file_path) as image:
            # convert image to readable data
            image = cv2.imread(file_path)
            # read meaningful data from image raw data
            text = pytesseract.image_to_string(image)
            print(text)
            return text
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""
# --------------------------------------------------------------------------------------------------------

# Function to read kitchen eatables
def read_kitchen_eatables():
    kitchen_items = []
    with open("Kitchen_Eatables_Database.txt", "r") as f:
        for line in f:
            kitchen_items.append(line.strip().lower())
    return kitchen_items  # Add this line to return the list
# --------------------------------------------------------------------------------------------------------

# Function to read non-food items
def nonfood_items_list():
    nonfood_items = []
    with open("NonFoodItems.txt", "r") as f:
        for line in f:
            nonfood_items.append(line.strip().lower())
    return nonfood_items  # Add this line to return the list
# --------------------------------------------------------------------------------------------------------

# Function to read irrelevant names
def irrelevant_names_list():
    irrelevant_names = []
    with open("Irrelevant.txt", "r") as file:
        for line in file:
            irrelevant_names.append(line.strip().lower())
    return irrelevant_names  # Add this line to return the list
# --------------------------------------------------------------------------------------------------------

# Function to add days to a date
def add_days(date_str, days_to_add):
    date_format = "%d/%m/%Y"
    date_obj = datetime.strptime(date_str, date_format)
    new_date = date_obj + timedelta(days=days_to_add)
    return new_date.strftime(date_format)
# --------------------------------------------------------------------------------------------------------

# Function to remove non apalphabet characters
def remove_non_alpha(substring):
    if any(c.isalpha() for c in substring):
        return re.sub(r"[^a-zA-Z]", " ", substring)
    else:
        return substring
# --------------------------------------------------------------------------------------------------------

# Function to contains_alphabet
def contains_alphabet(input_string):
    return any(char.isalpha() for char in input_string)
# --------------------------------------------------------------------------------------------------------

# Function to process string
def process_string(input_string):
    if contains_alphabet(input_string):
        return input_string
    else:
        return ""
# --------------------------------------------------------------------------------------------------------

# Function to add number if none
def add_number_if_none(string):
    if len(string) == 0 or not string[-1].isdigit():
        string += " 0"
    return string
# --------------------------------------------------------------------------------------------------------

# Function to process text
def process_text(text, kitchen_items, nonfood_items, irrelevant_names):
    # Creating a list of things split by new line
    lines = text.strip().split("\n")
    # Delete rows in list which are empty
    lines = [row for row in lines if row != ""]
    data_list = []
    for line in lines:
        # Ignore line which contain these words
        # Remove numbers embedded in name
        line = process_string(line)
        line = line.split()
        line = [remove_non_alpha(substring) for substring in line]
        line = " ".join(line)
        # Use regular expression to find the first occurrence of an alphabet
        match = re.search(r"[a-zA-Z]", line)
        if match:
            line = line[match.start() :]
        line = add_number_if_none(line)
        parts = line
        parts = parts.rsplit(maxsplit=1)
        if len(parts) < 2:
            continue
        name, price = parts
        try:
            data_list.append({"Item": name, "Price": price})
        except ValueError:
            continue
    #########################################################################
    # Deleting duplicate items
    df_new = pd.DataFrame(data_list)
    df_new2 = df_new.drop_duplicates(subset=["Item"])
    ##########################################################################
    # Remove any decimal/Floating number from item name
    # Check if the 'Item' column exists before processing
    if "Item" in df_new2.columns:
        df_new2.loc[:, "Item"] = df_new2["Item"].str.replace("\d+\.\d+\s", "")
        # Remove any number or character from item name. Remove starting or leading space.
        df_new2.loc[:, "Item"] = df_new2["Item"].str.replace("[^a-zA-Z\s]", " ")
        df_new2.loc[:, "Item"] = (
            df_new2["Item"].str.replace("[^a-zA-Z\s]", " ").str.strip()
        )
        df_new2 = df_new2[df_new2["Item"].str.strip() != ""]
        ##########################################################################
        # Create a boolean mask to identify rows where the 'Price' column contains '/'
        mask = df_new2["Price"].str.contains("/")
        # Invert the mask to get rows where 'Price' does not contain '/'
        df_new2 = df_new2[~mask]
        # Split the 'Name' column by spaces and join with single space
        df_new2["Item"] = df_new2["Item"].str.split().str.join(" ")
        # Filter the DataFrame to exclude rows with the specified conditions
        df_new2 = df_new2[
            ~(
                (
                    df_new2["Item"].str.contains("Date")
                    & df_new2["Item"].str.contains("pm")
                )
                | (
                    df_new2["Item"].str.contains("Date")
                    & df_new2["Item"].str.contains("am")
                )
                | (
                    df_new2["Item"].str.contains("Date")
                    & df_new2["Item"].str.contains("/")
                )
            )
        ]
        # Convert the names in the DataFrame to lowercase for case-insensitive comparison
        df_new2["Item"] = df_new2["Item"].str.lower()
        # Filter out rows with names that match those in "Irrelevant.txt" (case-insensitive)
        df_new2 = df_new2[~df_new2["Item"].isin(irrelevant_names)]
        # Convert the names in the DataFrame to uppercase
        df_new2["Item"] = df_new2["Item"].str.upper()
        # Filter out items with price greater than 500
        # ---------------------Start------------------------------
        # Remove non-numeric characters from 'Price' column
        df_new2["Price"] = df_new2["Price"].str.replace(r"[^0-9.]", "", regex=True)
        # Convert 'Price' column to numeric
        df_new2["Price"] = pd.to_numeric(
            df_new2["Price"], errors="coerce"
        )  # 'coerce' will handle any conversion errors
        # Replace price with 0 if > 500
        df_new2.loc[df_new2["Price"] > 500, "Price"] = 0
        # ----------------------Stop---------------------------------
        # Find date fromf user_{user_email}/Receipt and add it to dataframe
        # If unable to find the date add todays date
        # Add  todays date as new column in dateframe
        # Always uses day/month/year
        #
        date1 = str(search_dates(text))
        if date1 == "None":
            date1 = date.today()
            date1 = str(date1.strftime("%d/%m/%Y"))
            date_element = date1
            date_record.append(date_element)
        else:
            date1 = search_dates(text)
            # Fixing date
            # Make string from list element
            str1 = "".join(str(e) for e in date1)
            # Remove unwanted characters from date
            for char in str1:
                if char in "()'":
                    # Remove  ()'
                    str1 = str1.replace(char, "")
                    # print (str1)
            # Separating strings using , to collect xx\xx\xx formate of date
            date_list = str1.split(",")
            date_list_new = list()
            # -----------Start------------------------------------------------
            for x in date_list:
                if x.count("/") == 2 or x.count("-") == 2:
                    x = x.replace("-", "/")
                    date_list_new.append(x)
                    break
            if len(date_list_new) == 0:
                date1 = date.today()
                loc_date = str(date1.strftime("%d/%m/%Y"))
                date_list_new.append(loc_date)
            # -----------End------------------------------------------------
            # Get first set of string which represent date element properly
            date_element = date_list_new[0]
        ############################################################################
        # ----------------------Start--------------------------------------
        # Fix date format (new)
        # Remove leading space
        # Add following code
        date_element = date_element.strip()
        # Remove all characters after first space
        date_element = date_element.split(" ")[0]
        # Remove any space, non alphanumeric character except for "/"
        date_element = re.sub(r"[^a-zA-Z0-9/]", "", date_element)
        date_parts = date_element.split("/")
        day = date_parts[0]
        month = date_parts[1]
        year = date_parts[2]
        day = int(day)
        month = int(month)
        year1 = str(year)
        year = int(year)
        # Check and update day
        if day > 31:
            day = datetime.today().day
        # Check and update month
        if month > 12:
            month = datetime.today().month
        # Check and update month
        if year > 2100:
            year = datetime.today().year
        if len(year1) == 2:
            year = "20" + year1
        # Reformat the date
        date_element = f"{day}/{month}/{year}"
        # -----------------------------------------------------------------
        #############################################################################
        df_new2["Date"] = date_element
        #######################################################################
        # Continuous increment of index
        df_new2 = df_new2.reset_index(drop=True)
        ######################################################################
        df_new2 = df_new2.rename(columns={"Item": "Name"})
        #######################################################################
        # Code Removed
        #########################################################################
        # Add expiry date and status column
        # Upload expiry database
        expiry_df = pd.read_csv("items_expiry.txt", header=None, names=["Name", "Expiry"])
        df_new2["Expiry"] = df_new2["Name"].apply(
            lambda x: expiry_df[
                expiry_df["Name"].str.contains(x, case=False, regex=False)
            ]["Expiry"].values[0]
            if len(
                expiry_df[expiry_df["Name"].str.contains(x, case=False, regex=False)]
            )
            > 0
            else 0
        )
        df_new2["Expiry_Date"] = df_new2.apply(
            lambda row: add_days(row["Date"], row["Expiry"]), axis=1
        )
        df_new2 = df_new2.drop("Expiry", axis=1)
        df_new2 = df_new2.assign(Status="Not Expired")
        ##############################################################################
        with open("ItemCost.txt", "r") as file:
            item_costs = {}
            for line in file:
                item, cost = line.strip().rsplit(" ", 1)
                item_costs[item] = float(cost)
        # Iterate over List and Cost and update prices if they dont exist
        for index, row in df_new2.iterrows():
            item_name = row['Name']
            if row['Price'] == 0:  # Compare to 0, not '$0'
                if item_name in item_costs:
                    df_new2.at[index, 'Price'] = item_costs[item_name]  # Don't add '$'

        # Add $ sign to price if its missing
        # df_new2["Price"] = df_new2["Price"].apply(
        #     lambda x: "$" + str(x) if "$" not in str(x) else str(x)
        # )
        # ----------------------Start-------------------------
        # Add image URL column
        image_list = []
        default_image_url = "https://example.com/default_image.jpg"  # Replace with your desired default image URL
        for index, row in df_new2.iterrows():
            Name_temp = row["Name"]
            search_term = Name_temp
            url = f"https://www.google.com/search?q={search_term}&source=lnms&tbm=isch"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            img_links = soup.find_all("img")
            if len(img_links) > 1:
                image_list.append(img_links[1]["src"])
            else:
                image_list.append(default_image_url)
        df_new2["Image"] = image_list
        # Remove numeric characters from the 'Name' column
        df_new2["Name"] = df_new2["Name"].str.replace(r"\d+", "", regex=True)
        # --------------------end---------------------------------------------
        # ---------------------Start-------------------------
        # Deleting duplicate items
        df_new2 = df_new2.drop_duplicates(subset=["Name"])
        # ---------------------Stop-------------------------
        # Create empty dataframes for the kitchen and non-kitchen items
    df_kitchen = pd.DataFrame(columns=df_new2.columns)
    df_nonkitchen = pd.DataFrame(columns=df_new2.columns)
    # Iterate through each row in the original dataframe
    # Before comparison need to make both quantities lower case
    # comparing each word in df_new2 to each word in kitchen_words (new)
    # Overwrite the following code
    # Convert both item names and kitchen_items to lowercase and split into words
    # Convert both item names and non_Food_items to lowercase and split into words
    # Convert item names to lowercase and split into words
    for index, row in df_new2.iterrows():
        # Split the item name using either space or period as the delimiter and convert them to lowercase
        item_words = re.split(r'[ .]', row["Name"].lower())      
        # Initialize variables to keep track of the best match found so far
        best_match_score = 0
        best_kitchen_item = None     
        # Iterate through each kitchen item and calculate the match score
        for kitchen_item in kitchen_items:
            kitchen_words = [word.lower() for word in kitchen_item.split()]
            match_score = sum(word in kitchen_words for word in item_words)     
            # Update the best match if the current score is higher
            if match_score > best_match_score:
                best_match_score = match_score
                best_kitchen_item = kitchen_item  # Assign the best matching kitchen item
        # If there was a match found, append the row to df_kitchen with updated "Name" column
        if best_kitchen_item is not None:
            # Create a new row with the updated "Name" column
            updated_row = row.copy()
            updated_row["Name"] = best_kitchen_item          
            # Append the updated row to df_kitchen
            df_kitchen = df_kitchen._append(updated_row, ignore_index=True)
        else:
            # If no match was found. # Check each word in the item name against each word in non_Food_items
            non_Food_words = [
                word.lower() for phrase in nonfood_items for word in phrase.split()
            ]
            if any(word in non_Food_words for word in item_words):
                df_nonkitchen = df_nonkitchen._append(row)
    ##############################################################################
    # Create list of dictionary from kitchen and non kitchen dataframe
    # This helps in creating a .json file with correct
    data = []
    items_kitchen = df_kitchen.to_dict(orient="records")
    data = []
    items_nonkitchen = df_nonkitchen.to_dict(orient="records")
    item_frequency = {"Food": []}
    item_frequency = get_data_from_json("ItemsList", "item_frequency")
    # Append items_kitchen to the existing "Food" data
    if isinstance(item_frequency, str):
        item_frequency = json.loads(item_frequency)
    item_frequency.setdefault("Food", []).extend(items_kitchen)
    # if not isinstance(item_frequency, dict):
    #     item_frequency = {"Food": []}  # Initialize as dictionary if it is not
    # item_frequency.setdefault("Food", []).extend(items_kitchen)
    save_data_to_cloud_storage("ItemsList", "item_frequency", item_frequency)
    ##############################################################################
    ##############################################################################
    # When creating the final JSON output:
    result = {"Food": items_kitchen, "Not_Food": items_nonkitchen}
    for category in result:
        for item in result[category]:
            item['Price'] = f"${item['Price']:.2f}"  # Add '$' only for display
    return result
# --------------------------------------------------------------------------------------------------------

# Function to check frequency of the data
def check_frequency_function():  
    if not request.json or "condition" not in request.json:
        return jsonify({"error": "Invalid input. Please provide a 'condition'."}), 400
    choice = request.json.get("condition").lower()
    # Get the current date
    current_date = datetime.now()
    if choice == 'biweekly':
        # Check if it's a biweekly interval (every 14 days)
        if current_date.day % 14 == 0:
            execute_script = True
        else:
            execute_script = False
    elif choice == 'monthly':
        # Check if it's the last day of the month
        total_days_in_month = (current_date.replace(month=current_date.month % 12 + 1, day=1) - timedelta(days=1)).day
        if current_date.day == total_days_in_month:
            execute_script = True
        else:
            execute_script = False
    elif choice == 'today':
        # Check if it's today's date
        if current_date.day == current_date.day:
            execute_script = True
        else:
            execute_script = False
    else:
        print("Invalid choice. Please enter 'biweekly', 'monthly', or 'today'.")
        execute_script = False
    if execute_script:
        try:
            item_frequency_data = get_data_from_json("ItemsList", "item_frequency")
            # Check if the data is in string format, then parse it
            if isinstance(item_frequency_data, str):
                item_frequency_data = json.loads(item_frequency_data)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format in item frequency data."}), 500
        except Exception as e:
            return jsonify({"error": f"Failed to download item frequency data: {e}"}), 500   
        item_frequency = {}
        # Process the items
        for item in item_frequency_data.get("Food", []):
            item_name = item.get("Name")
            if item_name:
                item_frequency[item_name] = item_frequency.get(item_name, 0) + 1
        if item_frequency:
            sorted_item_frequency = dict(sorted(item_frequency.items(), key=lambda x: x[1], reverse=True))
            try:
                if not bucket_name:
                    return jsonify({"error": "BUCKET_NAME environment variable not set."}), 500
                save_data_to_cloud_storage("ItemsList", "item_frequency_sorted", sorted_item_frequency)
                save_data_to_cloud_storage("ItemsList", "item_frequency", json.dumps({"Food": []}))
            except Exception as e:
                return jsonify({"error": f"Failed to upload sorted item frequency data: {e}"}), 500
            return jsonify({
                "message": "Item frequency has been saved to item_frequency_sorted.json.",
                "sorted_item_frequency": sorted_item_frequency
            })
        else:
            return jsonify({"error": "No valid item data found in item_frequency.json."}), 400
    else:
        return jsonify({"message": "The script will not run."})
# --------------------------------------------------------------------------------------------------------

# Chat GPT Functions 
# --------------------------------------------------------------------------------------------------------
def food_handling_advice_using_gpt_function():
    try:
        # Fetch the content from storage
        content = get_data_from_json("ItemsList", "master_nonexpired")
        # Check if the content is a dictionary or a string
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        # If the content is a string, parse it into a dictionary
        if isinstance(content, str):
            food_handling_advice_data = json.loads(content)
        elif isinstance(content, dict):
            food_handling_advice_data = content
        else:
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if not isinstance(food_handling_advice_data, dict) or 'Food' not in food_handling_advice_data:
            raise ValueError("Invalid data format received from storage.")
        food_items = food_handling_advice_data['Food']
        food_items = [item for item in food_items if item['Name'] != 'TestFNE']
        # Define a list to store advice on handling food items
        food_handling_advice_list = []
        # Loop to generate advice for all food items
        for item in food_items:
            time.sleep(20)
            # Generate a prompt for GPT-3 to provide advice on handling food items
            prompt = f"Provide advice on how to handle {item['Name']} to increase its shelf life:"  
            # Use GPT-3 to generate advice
            response = openai.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=1000,
                temperature=0.6,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            handling_advice = response.choices[0].text.strip()
            food_handling_advice_list.append({
                "Food Item": item['Name'],
                "Handling Advice": handling_advice
            })
        # Save the updated advice list to cloud storage
        save_response = save_data_to_cloud_storage("ChatGPT/HomePage", "food_handling_advice", food_handling_advice_list)
        if save_response[1] != 200:
            raise Exception(save_response[0]['error'])
        return jsonify({"handlingadvice": food_handling_advice_list})
    except Exception as e:
        # Log the exception for debugging
        logging.exception("Exception occurred in food_handling_advice_using_gpt")
        return jsonify({"error": str(e)}), 500

def food_waste_reductions_using_gpt_function():
    try:
        # Extract specific input from the request if needed
        user_input = request.json.get("user_input", "Suggest a recipe that helps reduce food waste")  # Default to "general" if no input is provided
        food_waste_reduction_list = []
        num_suggestions = 1
        for _ in range(num_suggestions):
            time.sleep(20)
            prompt = f"{user_input}"
            response = openai.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=1000,
                temperature=0.6,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            food_waste_reduction_suggestion = response.choices[0].text.strip()
            food_waste_reduction_list.append({
                "Prompt": prompt,
                "Food Waste Reduction Suggestion": food_waste_reduction_suggestion,
            })  
        save_data_to_cloud_storage("ChatGPT/HomePage", "Food_Waste_Reduction_Suggestions", food_waste_reduction_list)    
        return jsonify({"foodWasteReductionSuggestions": food_waste_reduction_list})    
    except Exception as e:
        return jsonify({"error": str(e)}), 500    
    
def ethical_eating_suggestion_using_gpt_function():
    try:
        content = get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(content, bytes):
            content = content.decode('utf-8') # If the content is a string, parse it into a dictionary
        if isinstance(content, str):
            ethical_eating_list = json.loads(content)
        elif isinstance(content, dict):
            ethical_eating_list = content
        else:
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if not isinstance(ethical_eating_list, dict) or 'Food' not in ethical_eating_list:
            raise ValueError("Invalid data format received from storage.")
        food_items = ethical_eating_list['Food']
        ethical_eating_list = []
        num_prompts = 1
        for _ in range(num_prompts):
            group_of_items = [item['Name'] for item in food_items[:5]]
            prompt = 'Consider the ethical aspects of the following ingredients:\n\n'
            for item in group_of_items:
                prompt += f'- {item}\n'
            prompt = prompt.replace("- TestFNE\n", "")
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=300,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            ethical_suggestion = response.choices[0].text.strip()
            group_of_items = [item["Name"] for item in food_items if item["Name"] != "TestFNE"]
            ethical_eating_list.append({
                "Group of Items": group_of_items,
                "Ethical Eating Suggestions": ethical_suggestion
            })
        save_data_to_cloud_storage("ChatGPT/HomePage", "Ethical_Eating_Suggestions", ethical_eating_list)
        return jsonify({"ethicalEatingSuggestions": ethical_eating_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def get_fun_facts_using_gpt_function():
    try:
        # Retrieve and decode data from the storage
        content = get_data_from_json("ItemsList", "master_nonexpired")
        # Check if content is a dict or needs to be parsed
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        # If the content is a string, parse it into a dictionary
        if isinstance(content, str):
            Fun_Facts = json.loads(content)
        elif isinstance(content, dict):
            Fun_Facts = content
        else:
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if not isinstance(Fun_Facts, dict) or 'Food' not in Fun_Facts:
            raise ValueError("Invalid data format received from storage.")
        # Process the data
        food_items = Fun_Facts.get('Food', [])
        food_items = [item for item in food_items if item['Name'] != 'TestFNE']
        fun_facts = []
        num_fun_facts = 3
        for _ in range(num_fun_facts):
            selected_item = random.choice(food_items)
            prompt = f"Retrieve fascinating and appealing information about the following foods: {selected_item['Name']}. Include unique facts, health benefits, and any intriguing stories associated with each." 
            # Generate fun facts using GPT-3
            response = openai.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=500,
                temperature=0.6,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )  
            fun_fact = response.choices[0].text.strip()
            fun_facts.append({
                "Food Item": selected_item['Name'],
                "Fun Facts": fun_fact
            })
        # Save generated fun facts to cloud storage
        save_data_to_cloud_storage("ChatGPT/HomePage", "generated_fun_facts", fun_facts)  
        # Return the generated fun facts
        return jsonify({"funFacts": fun_facts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def cooking_tips_using_gpt_function():
    try:
        Cooking_Tips_List = []
        # Define the number of tips you want to generate
        num_tips = 3
        # Loop to generate multiple cooking tips
        for _ in range(num_tips):
            # Introduce randomness in the prompt
            prompt = f"Seek advice on {random.choice(['cooking techniques', 'tips for improving a dish', 'alternative ingredients for dietary restrictions'])}."
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=500,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            tip = response.choices[0].text.strip()
            Cooking_Tips_List.append({"Prompt": prompt, "Cooking Tip": tip})
            save_data_to_cloud_storage("ChatGPT/HomePage", "Cooking_Tips", Cooking_Tips_List)
            return jsonify({"cookingTips": Cooking_Tips_List})
    except Exception as e:
        return jsonify({"error": str(e)})

def current_trends_using_gpt_function():
    try:
        # Set up client API
        # Define a list to store fun facts
        fun_facts = []
        # Define the number of fun facts you want to generate
        num_fun_facts = 1
        # Loop to generate multiple fun facts about food trends and innovations
        for _ in range(num_fun_facts):
            # Introduce randomness in the prompt
            prompt = f"Stay updated on {random.choice(['exciting', 'cutting-edge', 'latest'])} food trends, {random.choice(['innovations', 'revolutions', 'breakthroughs'])}, or {random.choice(['unique', 'extraordinary', 'exceptional'])} culinary experiences. Provide youtube channels, blogs, twitter groups."
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=300,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            fun_fact = response.choices[0].text.strip()
            fun_facts.append({"Prompt": prompt, "Fun Facts": fun_fact})
        save_data_to_cloud_storage("ChatGPT/HomePage", "Current_Trends", fun_facts)
        return jsonify({"currentTrends": fun_facts})
    except Exception as e:
        return jsonify({"error": str(e)})

def mood_changer_using_gpt_function():
    try:
        user_mood = request.json.get("user_mood", "Sad, I'm feeling tired, I'm going to bed")
        # Set up client API
        # Define a list to store mood-based food suggestions
        food_suggestions_list = []
        # Define the number of suggestions you want to generate
        num_suggestions = 1
        # Loop to generate mood-based food suggestions
        for _ in range(num_suggestions):
            time.sleep(20)
            # Introduce user mood in the prompt
            prompt = (
                f"Suggest a food that can improve my mood when I'm feeling {user_mood}."
            )
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=300,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            food_suggestion = response.choices[0].text.strip()
            food_suggestions_list.append(
                {
                    "User Mood": user_mood,
                    "Prompt": prompt,
                    "Food Suggestion": food_suggestion,
                }
            )
        save_data_to_cloud_storage("ChatGPT/HomePage", "Mood_Changer", food_suggestions_list)
        return jsonify({"moodChangerSuggestions": food_suggestions_list})
    except Exception as e:
        return jsonify({"error": str(e)})

def jokes_using_gpt_function():
    try:
        # Define a list to store food-related jokes
        jokes_list = []
        # Define a list of prompts for random jokes
        prompts = [
            "Tell me a random joke of the day with a food-related theme.",
            "Give me a funny joke about food.",
            "What's a hilarious food-related joke?",
            "Share a random food joke with me.",
            "Tell a joke about food that's sure to make me laugh."
        ]
        # Define the number of jokes you want to generate
        num_jokes = 1
        # Loop to generate multiple food-related jokes
        for _ in range(num_jokes):
            time.sleep(20)  # To avoid rate limits, adjust as needed
            # Randomly select a prompt from the list
            prompt = random.choice(prompts)
            response = openai.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=300,
                temperature=0.6,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            joke = response.choices[0].text.strip()
            jokes_list.append({"Prompt": prompt, "Food Joke": joke})   
        # Save the jokes to cloud storage
        save_data_to_cloud_storage("ChatGPT/HomePage", "Joke", jokes_list)
        return jsonify({"jokes": jokes_list})
    except Exception as e:
        return jsonify({"error": str(e)})
    
def nutritional_value_using_gpt_function():
    try:
        # Load data from JSON
        content = get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        if isinstance(content, str):
            content = json.loads(content)
        elif not isinstance(content, dict):
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if 'Food' not in content:
            raise ValueError("Invalid data format received from storage.")
        # Access the food items list
        food_items = [item for item in content['Food'] if item['Name'] != 'TestFNE']
        # Set up client API (assume OpenAI API is already set up elsewhere in your code)
        # Define a list to store nutritional advice
        nutritional_advice = []
        # Define the number of advice you want to generate
        num_advice = 5
        # Loop to generate multiple advice
        for _ in range(num_advice):
            time.sleep(20)
            # Randomly select a food item
            selected_item = random.choice(food_items)
            prompt = f"Provide nutritional advice for incorporating {selected_item['Name']} into a balanced diet:"
            response = openai.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=1000,
                temperature=0.6,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            advice = response.choices[0].text.strip()
            nutritional_advice.append({
                "Food Item": selected_item['Name'],
                "Nutritional Advice": advice
            })
        # Save the generated advice back to the cloud storage
        save_data_to_cloud_storage("ChatGPT/Health", "generated_nutritional_advice", nutritional_advice)
        return jsonify({"nutritionalValue": nutritional_advice})
    except Exception as e:
        return jsonify({"error": str(e)})

def allergy_information_using_gpt_function():
    try:
        # Load data from JSON
        content = get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        # If the content is a string, parse it into a dictionary
        if isinstance(content, str):
            content = json.loads(content)
        elif isinstance(content, dict):
            content = content
        else:
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if not isinstance(content, dict) or 'Food' not in content:
            raise ValueError("Invalid data format received from storage.")  
        food_items = content['Food']
        food_items = [item for item in food_items if item['Name'] != 'TestFNE']
        # Set up client API
        # Define a list to store allergy-related information for specific food items
        allergy_information_list = []
        # Define the number of allergy-related prompts you want to generate
        num_prompts = 3
        # Loop to generate allergy-related information for all food items
        for index, item in enumerate(food_items):
            time.sleep(20)
            if index >=4:
                break
            # Generate allergy-related prompt
            allergy_prompt = f"Allergy side effects of {item['Name']}:"
            response_allergy = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=allergy_prompt,
            max_tokens=1000,  # Adjust the value based on your needs
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            allergy_information = response_allergy.choices[0].text.strip()
            allergy_information_list.append(
                {"Food Item": item["Name"], "Allergy Information": allergy_information}
            )
        save_data_to_cloud_storage("ChatGPT/Health", "allergy_information", allergy_information_list)
        return jsonify({"AllergyInformation": allergy_information_list})
    except Exception as e:
        return jsonify({"error": str(e)})

def healthier_alternatives_using_gpt_function():
    try:
        # Load data from JSON
        content = get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        # If the content is a string, parse it into a dictionary
        if isinstance(content, str):
            content = json.loads(content)
        elif isinstance(content, dict):
            content = content
        else:
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if not isinstance(content, dict) or 'Food' not in content:
            raise ValueError("Invalid data format received from storage.") 
        food_items = content['Food']
        food_items = [item for item in food_items if item['Name'] != 'TestFNE']
        # Set up client API
        # Define a list to store suggestions and cheaper alternatives for specific food items
        food_suggestions_with_alternatives = []
        # Define the number of suggestions you want to generate
        num_suggestions = 3
        # Loop to generate suggestions and cheaper alternatives for all food items
        for item in food_items:
            time.sleep(20)
            # Generate suggestion
            suggestion_prompt = (
                f"Suggest ways to incorporate {item['Name']} into a healthy diet:"
            )
            response_suggestion = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=suggestion_prompt,
            max_tokens=1000,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            # Generate cheaper alternative
            cheaper_alternative_prompt = (
                f"Suggest a healthier alternative to {item['Name']}:"
            )
            response_alternative = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=cheaper_alternative_prompt,
            max_tokens=1000,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            cheaper_alternative = response_alternative.choices[0].text.strip()
            food_suggestions_with_alternatives.append(
                {"Food Item": item["Name"], "Healthy Alternative": cheaper_alternative}
            )
            save_data_to_cloud_storage("ChatGPT/Health", "Healthy_alternatives", food_suggestions_with_alternatives)
            return jsonify({"alternatives": food_suggestions_with_alternatives})
    except Exception as e:
        return jsonify({"error": str(e)})

def healthy_eating_advice_using_gpt_function():
    try:
        # Set up client AP
        # Define a list to store eating advice-related information
        eating_advice_list = []
        # Define the number of prompts you want to generate
        num_prompts = 1
        # Loop to generate eating advice for the specified number of prompts
        for _ in range(num_prompts):
            time.sleep(20)
            # Generate eating advice prompt
            eating_advice_prompt = "Provide general advice for maintaining healthy eating habits:"
            response_eating_advice = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=eating_advice_prompt,
            max_tokens=500,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)   
            eating_advice_response = response_eating_advice.choices[0].text.strip()
            # Remove alphanumeric characters using regex
            eating_advice_response = re.sub(r"[^a-zA-Z\s]", "", eating_advice_response)
            eating_advice_list.append({"Prompt": eating_advice_prompt, "Health Advice": eating_advice_response})
        save_data_to_cloud_storage("ChatGPT/Health", "healthy_eating_advice", eating_advice_list)
        return jsonify({"eatingAdviceList": eating_advice_list})
    except Exception as e:
        return jsonify({"error": str(e)})

def health_advice_using_gpt_function():
    try:
        # Set up client API
        # Define a list to store health and diet advice
        Health_Advice_List = []
        # Define the number of advice you want to generate
        num_advice = 3
        # Loop to generate multiple pieces of health and diet advice
        for _ in range(num_advice):
            # Introduce randomness in the prompt
            prompt = f"Get general information or tips on {random.choice(['healthy eating', 'dietary plans', 'specific nutritional topics'])}."
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=300,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            advice = response.choices[0].text.strip()
            Health_Advice_List.append({"Prompt": prompt, "Health Advice": advice})
            save_data_to_cloud_storage("ChatGPT/Health", "Health_Advice", Health_Advice_List)
            return jsonify({"healthAdviceList": Health_Advice_List})
    except Exception as e:
        return jsonify({"error": str(e)})

def healthy_items_usage_using_gpt_function():
    try:
        # Load data from JSON
        content = get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        # If the content is a string, parse it into a dictionary
        if isinstance(content, str):
            content = json.loads(content)
        elif isinstance(content, dict):
            content = content
        else:
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if not isinstance(content, dict) or 'Food' not in content:
            raise ValueError("Invalid data format received from storage.") 
        food_items = content['Food']
        food_items = [item for item in food_items if item['Name'] != 'TestFNE']
        # Set up client API
        # Define a list to store suggestions for specific food items
        specific_food_suggestions = []
        # Define the number of suggestions you want to generate
        num_suggestions = 3
        # Loop to generate suggestions for all food items
        for item in food_items:
            prompt = f"Suggest ways to incorporate {item['Name']} into a healthy diet:"
            time.sleep(20)
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=3000,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            suggestion = response.choices[0].text.strip()
            specific_food_suggestions.append(
                {"Food Item": item["Name"], "Suggestion": suggestion}
            )
        save_data_to_cloud_storage("ChatGPT/Health", "healthy_usage", specific_food_suggestions)
        return jsonify({"suggestions": specific_food_suggestions})
    except Exception as e:
        return jsonify({"error": str(e)})

def nutritional_analysis_using_gpt_function():
    try:
        # Load data from JSON
        content = get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        # If the content is a string, parse it into a dictionary
        if isinstance(content, str):
            content = json.loads(content)
        elif isinstance(content, dict):
            content = content
        else:
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if not isinstance(content, dict) or 'Food' not in content:
            raise ValueError("Invalid data format received from storage.")
        food_items = content['Food']
        food_items = [item for item in food_items if item['Name'] != 'TestFNE']
        # Set up client API
        # Define a list to store mood-based food suggestions
        food_suggestions_list = []
        # Define the number of suggestions you want to generate
        num_suggestions = 1
        # Loop to generate mood-based food suggestions
        for _ in range(num_suggestions):
            group_of_items = [
                item["Name"] for item in food_items[:5]
            ]  # Change the slicing as needed
            prompt = "Generate a nutritional analysis. Mention which part of healthy diet is missing. Suggest new items to fill the gaps for the following ingredients:\n\n"
            for item in group_of_items:
                prompt += f"- {item}\n"
            # Remove "- TestFNE" from the prompt
            prompt = prompt.replace("- TestFNE\n", "")
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=300,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            analysis = response.choices[0].text.strip()
            # Extract a group of items, excluding 'TestFNE'
            group_of_items = [
                item["Name"] for item in food_items if item["Name"] != "TestFNE"
            ]
            food_suggestions_list.append(
                {"Group of Items": group_of_items, "Nutritional Analysis": analysis}
            )
            save_data_to_cloud_storage("ChatGPT/Health", "Nutritional_Analysis", food_suggestions_list)
            return jsonify({"nutritionalAnalysis": food_suggestions_list})
    except Exception as e:
        return jsonify({"error": str(e)})
    
def health_incompatibilities_using_gpt_function():
    try:
        # Load data from JSON
        content = get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        # If the content is a string, parse it into a dictionary
        if isinstance(content, str):
            content = json.loads(content)
        elif isinstance(content, dict):
            content = content
        else:
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if not isinstance(content, dict) or 'Food' not in content:
            raise ValueError("Invalid data format received from storage.")
        food_items = content['Food']
        food_items = [item for item in food_items if item['Name'] != 'TestFNE']
        # Set up client API
        # Combine all food item names into a single prompt
        food_names_combined = ", ".join([item['Name'] for item in food_items])
        # Define a list to store health-wise incompatibility information for all food items together
        incompatibility_information_list = []
        # Generate a health-wise incompatibility prompt for all food items together
        incompatibility_prompt = f"Check for health-wise incompatibility of consuming {food_names_combined} together:"    
        response_incompatibility = openai.completions.create(model="gpt-3.5-turbo-instruct",
        prompt=incompatibility_prompt,
        max_tokens=500,  # Adjust max_tokens based on your needs
        temperature=0.6,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0)
        incompatibility_information = response_incompatibility.choices[0].text.strip()
        group_of_items = [
                item["Name"] for item in food_items if item["Name"] != "TestFNE"
            ]
        incompatibility_information_list.append({
            "Food Combination": group_of_items,
            "Health-wise Incompatibility Information": incompatibility_information
        })
        save_data_to_cloud_storage("ChatGPT/Health", "health_incompatibility_information_all", incompatibility_information_list)
        return jsonify({"healthIncompatibilities": incompatibility_information_list})
    except Exception as e:
        return jsonify({"error": str(e)})

def user_defined_dish_using_gpt_function():
    try:
        user_dish = request.json.get("user_dish", "Sweet Dish")
        # Set up client API
        # Define a list to store fun facts
        fun_facts = []
        # Define the number of fun facts you want to generate
        num_fun_facts = 1
        # Loop to generate multiple fun facts about food trends and innovations
        for _ in range(num_fun_facts):
            time.sleep(20)
            # Introduce randomness in the prompt
            prompt = f"Create food recipe for {user_dish}"
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=1000,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            fun_fact = response.choices[0].text.strip()
            fun_facts.append({"Prompt": prompt, "Fun Facts": fun_fact})
        save_data_to_cloud_storage("ChatGPT/Recipe", "User_Defined_Dish", fun_facts)
        return jsonify({"definedDishes": fun_facts})
    except Exception as e:
        return jsonify({"error": str(e)})

def fusion_cuisine_using_gpt_function():
    try:
        user_input = request.json.get("user_input", "Italian and Japanese")
        # Set up client API
        # Define a list to store fusion cuisine suggestions
        fusion_suggestions_list = []
        # Define the number of suggestions you want to generate
        num_suggestions = 1
        # Loop to generate fusion cuisine suggestions
        for _ in range(num_suggestions):
            time.sleep(20)
            # Introduce user input in the prompt
            prompt = f"Suggest a fusion cuisine that combines {user_input} flavors."
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=300,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            fusion_suggestion = response.choices[0].text.strip()
            fusion_suggestions_list.append(
                {
                    "User Input": user_input,
                    "Prompt": prompt,
                    "Fusion Cuisine Suggestion": fusion_suggestion,
                }
            )   
            save_data_to_cloud_storage("ChatGPT/Recipe", "Fusion_Cuisine_Suggestions", fusion_suggestions_list)
            return jsonify({"fusionSuggestions": fusion_suggestions_list})
    except Exception as e:
        return jsonify({"error": str(e)})

def unique_recipes_using_gpt_function():
    try:
        # Define a list to store user-specific ecipes
        unique_recipe = request.json.get("unique_recipe", "banana rice apple")
        user_recipes_list = []
        # Define the number of recipes you want to generate
        num_recipes = 1
        # Loop to generate user-specific recipes
        for _ in range(num_recipes):
            # Introduce user input in the prompt
            prompt = f"Create a unique recipe based on the user input: {unique_recipe}."
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=500,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            recipe = response.choices[0].text.strip()
            # Generate a random encouraging remark focused on future efforts
            future_encouragement = [
                "Keep exploring new recipes in the future!",
                "Looking forward to your next culinary adventure!",
                "You're on a cooking journey - exciting times ahead!",
                "Imagine the delicious recipes you'll discover in the future!",
            ]
            random_encouragement = random.choice(future_encouragement)
            user_recipes_list.append(
                {
                    "User Input": unique_recipe,
                    "Prompt": prompt,
                    "Recipe": recipe,
                    "Encouragement": random_encouragement,
                }
            ) 
            save_data_to_cloud_storage("ChatGPT/Recipe", "Unique_Recipes", user_recipes_list)
            return jsonify({"uniqueRecipes": user_recipes_list})
    except Exception as e:
        return jsonify({"error": str(e)})
    
def recipes_using_gpt_function():
    try:
        # Load data from JSON
        content = get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        # If the content is a string, parse it into a dictionary
        if isinstance(content, str):
            content = json.loads(content)
        elif isinstance(content, dict):
            content = content
        else:
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if not isinstance(content, dict) or 'Food' not in content:
            raise ValueError("Invalid data format received from storage.")
        food_items = content['Food']
        food_items = [item for item in food_items if item['Name'] != 'TestFNE']
        # Set up client API
        # Define a list to store recipes
        recipes = []
        # Define the number of recipes you want to generate
        num_recipes = 3
        # Loop to generate multiple recipes
        for _ in range(num_recipes):
            # Extract a group of items from the food_items list (for example, first 5 items)
            group_of_items = [
                item["Name"] for item in food_items[:5]
            ]  # Change the slicing as needed
            prompt = "Generate a recipe using the following ingredients:\n\n"
            for item in group_of_items:
                prompt += f"- {item}\n"
            # Remove "- TestFNE" from the prompt
            prompt = prompt.replace("- TestFNE\n", "")
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=500,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            recipe = response.choices[0].text.strip()
            recipe = recipe.replace("\n", " ")
            recipe = recipe.split("Instructions:")[1].strip()
            # Extract a group of items, excluding 'TestFNE'
            group_of_items = [
                item["Name"] for item in food_items if item["Name"] != "LARGE EGGS"
            ]
            recipes.append({"Group of Items": group_of_items, "Generated Recipe": recipe})
            save_data_to_cloud_storage("ChatGPT/Recipe", "generated_recipes", recipes)
            return jsonify({"generatedRecipes": recipes})
    except Exception as e:
        return jsonify({"error": str(e)})
    
def diet_schedule_using_gpt_function():
    try:
        # load data from JSON
        content = get_data_from_json("ItemsList", "master_nonexpired")
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        # If the content is a string, parse it into a dictionary
        if isinstance(content, str):
            content = json.loads(content)
        elif isinstance(content, dict):
            content = content
        else:
            raise TypeError("Unexpected content type returned by get_data_from_json")
        # Check if the data is a dictionary and contains the 'Food' key
        if not isinstance(content, dict) or 'Food' not in content:
            raise ValueError("Invalid data format received from storage.")
        food_items = content['Food']
        food_items = [item for item in food_items if item['Name'] != 'TestFNE']
        # Set up client API
        # Define a list to store the diet schedule
        diet_schedule = []
        # Define the number of meals in the diet schedule
        num_meals = 5
        # Define meal categories
        meal_categories = ["breakfast", "snack", "lunch", "snack", "dinner"]
        # Loop to generate a diet schedule with specified number of meals
        for meal_number in range(1, num_meals + 1):
            time.sleep(20)
            # Randomly select a food item for each meal
            selected_item = random.choice(food_items)
            # Get the meal category for the current meal number
            meal_category = meal_categories[meal_number - 1]
            # Generate a prompt for GPT-3 to provide a meal suggestion
            prompt = f"Create a {meal_category} suggestion for meal {meal_number} using {selected_item['Name']} and other healthy ingredients:"
            # Use GPT-3 to generate a meal suggestion
            response = openai.completions.create(model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=500,
            temperature=0.6,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)
            meal_suggestion = response.choices[0].text.strip()
            diet_schedule.append(
                {
                    "Meal Number": meal_number,
                    "Meal Category": meal_category,
                    "Food Item": selected_item["Name"],
                    "Meal Suggestion": meal_suggestion,
                }
            )  
            save_data_to_cloud_storage("ChatGPT/Recipe", "diet_schedule", diet_schedule)
            return jsonify({"dietSchedule": diet_schedule})
    except Exception as e:
        return jsonify({"error": str(e)})
# --------------------------------------------------------------------------------------------------------

##############################################################################################################################################################################
##############################################################################################################################################################################                                              
                                                
                                                # ChatGpt Prompts End Point Apis
# Homepage (cooking_tips, current_trends, ethical_eating_suggestions, food_waste_reductions,generated_func_facts, joke, mood_changer)
##############################################################################################################################################################################
@app.route("/api/food-handling-advice-using-json", methods=["GET"])
@authenticate_user_function
def food_handling_advice_using_json():
    try:
        data = get_data_from_json("ChatGPT/HomePage", "food_handling_advice")
        if "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"handlingadvice": data}), 200
    except Exception as e:
        logging.exception("Exception occurred in food_handling_advice_using_json")
        return jsonify({"error": str(e)}), 500
@app.route("/api/food-handling-advice-using-gpt", methods=["POST", "GET"])
@authenticate_user_function
def food_handling_advice_using_gpt():
    return food_handling_advice_using_gpt_function()

@app.route("/api/food-waste-reduction-using-json", methods=["GET"])
@authenticate_user_function
def food_waste_reduction_using_json():
    try:
        data = get_data_from_json("ChatGPT/HomePage", "Food_Waste_Reduction_Suggestions")
        if "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"foodWasteReductionSuggestions": data}), 200
    except Exception as e:
        logging.exception("Exception occurred in food_waste_reduction_using_json")
        return jsonify({"error": str(e)}), 500

@app.route("/api/food-waste-reduction-using-gpt", methods=["POST"])
@authenticate_user_function
def food_waste_reduction_using_gpt():
    return food_waste_reductions_using_gpt_function()

@app.route("/api/ethical-eating-suggestion-using-json", methods=["GET"])
@authenticate_user_function
def ethical_eating_using_json():
    try:
        data = get_data_from_json("ChatGPT/HomePage", "Ethical_Eating_Suggestions")
        if "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"ethicalEatingSuggestions": data}), 200
    except Exception as e:
        logging.exception("Exception occurred in ethical_eating_using_json")
        return jsonify({"error": str(e)}), 500
@app.route("/api/ethical-eating-suggestion-using-gpt", methods=["POST", "GET"])
@authenticate_user_function
def ethical_eating_suggestion_using_gpt():
    return ethical_eating_suggestion_using_gpt_function()

@app.route("/api/get-fun-facts-using-json", methods=["GET"])
@authenticate_user_function
def get_fun_facts_using_json():
    try:
        data = get_data_from_json("ChatGPT/HomePage", "generated_fun_facts")
        if "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"funFacts": data}), 200
    except Exception as e:
        logging.exception("Exception occurred in get_fun_facts_using_json")
        return jsonify({"error": str(e)}), 500
@app.route("/api/get-fun-facts-using-gpt", methods=["POST", "GET"])
@authenticate_user_function
def get_fun_facts_using_gpt():
    return get_fun_facts_using_gpt_function()

@app.route("/api/cooking-tips-using-json", methods=["GET"])
@authenticate_user_function
def cooking_tips_using_json():
    try:
        data = get_data_from_json("ChatGPT/HomePage", "Cooking_Tips")
        if "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"cookingTips": data}), 200
    except Exception as e:
        logging.exception("Exception occurred in cooking_tips_using_json")
        return jsonify({"error": str(e)}), 500
@app.route("/api/cooking-tips-using-gpt", methods=["POST", "GET"])
@authenticate_user_function
def cooking_tips_using_gpt():
    return cooking_tips_using_gpt_function()

@app.route("/api/current-trends-using-json", methods=["GET"])
@authenticate_user_function
def current_trends_using_json():
    try:
        data = get_data_from_json("ChatGPT/HomePage", "Current_Trends")
        if "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"currentTrends": data}), 200
    except Exception as e:
        logging.exception("Exception occurred in current_trends_using_json")
        return jsonify({"error": str(e)}), 500
@app.route("/api/current-trends-using-gpt", methods=["POST", "GET"])
@authenticate_user_function
def current_trends_using_gpt():
    return current_trends_using_gpt_function()

@app.route("/api/mood-changer-using-json", methods=["GET"])
@authenticate_user_function
def mood_changer_using_json():
    try:
        data = get_data_from_json("ChatGPT/HomePage", "Mood_Changer")
        if "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"moodChangerSuggestions": data}), 200
    except Exception as e:
        logging.exception("Exception occurred in mood_changer_using_json")
        return jsonify({"error": str(e)}), 500
@app.route("/api/mood-changer-using-gpt", methods=["POST", "GET"])
@authenticate_user_function
def mood_changer_using_gpt():
    return mood_changer_using_gpt_function()

@app.route("/api/jokes-using-json", methods=["GET"])
@authenticate_user_function
def jokes_json():
    try:
        data = get_data_from_json("ChatGPT/HomePage", "Joke")
        if "error" in data:
            return jsonify({"error": data["error"]}), 500
        return jsonify({"jokes": data}), 200
    except Exception as e:
        logging.exception("Exception occurred in jokes_json")
        return jsonify({"error": str(e)}), 500
@app.route("/api/jokes-using-gpt", methods=["POST", "GET"])
@authenticate_user_function
def jokes_using_gpt():
    return jokes_using_gpt_function()


# Health and Diet Advice (allergy_information, food_handling_advice, generated_nutrition_advice, health_advice,health_incompatibility_information, health_alternatives, healthy_eating_advice, healthy_usage, nutritional_analysis, nutritioanl_value)
##############################################################################################################################################################################
@app.route("/api/nutritional-value-using-json", methods=["GET"])
@authenticate_user_function
def nutritional_value_using_json():
    data = get_data_from_json("ChatGPT/Health", "generated_nutritional_advice")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"nutritionalValue": data})
@app.route("/api/nutritional-value-using-gpt", methods=["GET", "POST"])
@authenticate_user_function
def nutritional_value_using_gpt():
    return nutritional_value_using_gpt_function()

@app.route("/api/allergy-information-using-json", methods=["GET"])
@authenticate_user_function
def allergy_information_using_json():
    data = get_data_from_json("ChatGPT/Health", "allergy_information")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"AllergyInformation": data})
@app.route("/api/allergy-information-using-gpt", methods=["GET", "POST"])
@authenticate_user_function
def allergy_information_using_gpt():
    return allergy_information_using_gpt_function()

@app.route("/api/healthier-alternatives-using-json", methods=["GET"])
@authenticate_user_function
def healthier_alternatives_using_json():
    data = get_data_from_json("ChatGPT/Health", "Healthy_alternatives")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"alternatives": data})
@app.route("/api/healthier-alternatives-using-gpt", methods=["GET", "POST"])
@authenticate_user_function
def healthier_alternatives_using_gpt():
    return healthier_alternatives_using_gpt_function()

@app.route("/api/healthy-eating-advice-using-json", methods=["GET"])
@authenticate_user_function
def healthy_eating_advice_using_json():
    data = get_data_from_json("ChatGPT/Health", "healthy_eating_advice")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"eatingAdviceList": data})
@app.route("/api/healthy-eating-advice-using-gpt", methods=["GET", "POST"])
@authenticate_user_function
def healthy_eating_advice_using_gpt():
    return healthy_eating_advice_using_gpt_function()

@app.route("/api/health-advice-using-json", methods=["GET"])
@authenticate_user_function
def health_advice_using_json():
    data = get_data_from_json("ChatGPT/Health", "Health_Advice")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"healthAdviceList": data})
@app.route("/api/health-advice-using-gpt", methods=["GET", "POST"])
@authenticate_user_function
def health_advice_using_gpt():
    return health_advice_using_gpt_function()

@app.route("/api/healthy-items-usage-using-json", methods=["GET"])
@authenticate_user_function
def healthy_items_usage_using_json():
    data = get_data_from_json("ChatGPT/Health", "healthy_usage")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"suggestions": data})
@app.route("/api/healthy-items-usage-using-gpt", methods=["GET", "POST"])
@authenticate_user_function
def healthy_items_usage():
    return healthy_items_usage_using_gpt_function()

@app.route("/api/nutritional-analysis-using-json", methods=["GET"])
@authenticate_user_function
def nutritional_analysis_using_json():
    data = get_data_from_json("ChatGPT/Health", "Nutritional_Analysis")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"nutritionalAnalysis": data})
@app.route("/api/nutritional-analysis-using-gpt", methods=["GET", "POST"])
@authenticate_user_function
def nutritional_analysis_using_gpt():
    return nutritional_analysis_using_gpt_function()

@app.route("/api/health_incompatibilities_using_json", methods=["GET"])
@authenticate_user_function
def health_incompatibilities_using_json():
    data = get_data_from_json("ChatGPT/Health", "health_incompatibility_information_all")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"healthIncompatibilities": data})

@app.route("/api/health_incompatibilities_using_gpt", methods=["GET", "POST"])
@authenticate_user_function
def health_incompatibilities_using_gpt():
    return health_incompatibilities_using_gpt_function()

# Recipe ( Cheap_alternatives, diet_schedule, fusion_cuisine_suggestion, generated_recipes, unique_recipes, user_defined_dish )
##############################################################################################################################################################################
@app.route("/api/user-defined-dish-using-json", methods=["GET"])
@authenticate_user_function
def user_defined_dish_using_json():
    data = get_data_from_json("ChatGPT/Recipe", "User_Defined_Dish")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"definedDishes": data})
@app.route("/api/user-defined-dish-using-gpt", methods=["GET", "POST"])
@authenticate_user_function
def user_defined_dish():
    return user_defined_dish_using_gpt_function()

@app.route("/api/fusion-cuisine-suggestions-using-json", methods=["GET"])
@authenticate_user_function
def fusion_cuisine_suggestions_using_json():
    data = get_data_from_json("ChatGPT/Recipe", "Fusion_Cuisine_Suggestions")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"fusionSuggestions": data})
@app.route("/api/fusion-cuisine-suggestion-using-gpt", methods=["GET", "POST"])
@authenticate_user_function
def fusion_cuisine_using_gpt():
    return fusion_cuisine_using_gpt_function()

@app.route("/api/unique-recipes-using-json", methods=["GET"])
@authenticate_user_function
def unique_recipes_using_json():
    data = get_data_from_json("ChatGPT/Recipe", "Unique_Recipes")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"uniqueRecipes": data})
@app.route("/api/unique-recipes-using-gpt", methods=["POST", "GET"])
@authenticate_user_function
def unique_recipes_using_gpt():
    return unique_recipes_using_gpt_function()

@app.route("/api/recipes-using-json", methods=["GET"])
@authenticate_user_function
def recipes_using_json():
    data = get_data_from_json("ChatGPT/Recipe", "generated_recipes")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"generatedRecipes": data})
@app.route("/api/recipes-using-gpt", methods=["POST", "GET"])
@authenticate_user_function
def recipes_using_gpt():
    return recipes_using_gpt_function()

@app.route("/api/diet-schedule-using-json", methods=["GET"])
@authenticate_user_function
def diet_schedule_using_json():
    data = get_data_from_json("ChatGPT/Recipe", "diet_schedule")
    if "error" in data:
        return jsonify({"error": data["error"]}), 500
    return jsonify({"dietSchedule": data})
@app.route("/api/diet-schedule-using-gpt", methods=["POST", "GET"])
@authenticate_user_function
def diet_schedule_using_gpt():
    return diet_schedule_using_gpt_function()
##############################################################################################################################################################################
##############################################################################################################################################################################
 
                                                    # Main Code End Point Apis
##############################################################################################################################################################################
# Delete all Items
@app.route("/api/deleteAll/master-nonexpired", methods=["POST"])
@authenticate_user_function
def deleteAll_master_nonexpired():
    return delete_all_items("master_nonexpired")

@app.route("/api/deleteAll/master-expired", methods=["POST"])
@authenticate_user_function
def deleteAll_master_expired():
    return delete_all_items("master_expired")

@app.route("/api/deleteAll/shopping-list", methods=["POST"])
@authenticate_user_function
def deleteAll_shopping():
    return delete_all_items("shopping_list")

@app.route("/api/deleteAll/purchase-list", methods=["POST"])
@authenticate_user_function
def deleteAll_purchase():
    return delete_all_items("result")
##############################################################################################################################################################################

# Add Custom Items
@app.route("/api/add-custom-item", methods=["POST"])
@authenticate_user_function
def add_custom_item():
    return add_custom_item_function()
##############################################################################################################################################################################

# Update Expiry
@app.route("/api/update-master-nonexpired-item-expiry", methods=["POST"])
@authenticate_user_function
def update_master_nonexpired_item_expiry():
    return update_master_nonexpired_item_expiry_function()
##############################################################################################################################################################################

# Get List of master_expired master_nonexpired and shopping_list
@app.route("/api/get-master-expired-list", methods=["GET"])
@authenticate_user_function
def get_master_expired():
    return get_file_response_base64("master_expired")

@app.route("/api/get-shopping-list", methods=["GET"])
@authenticate_user_function
def get_shopping_list():
    return get_file_response_base64("shopping_list")

@app.route("/api/get-master-nonexpired-list", methods=["GET"])
@authenticate_user_function
def get_master_nonexpired():
    return get_file_response_base64("master_nonexpired")

@app.route("/api/get-purchased-list", methods=["GET"])
@authenticate_user_function
def get_purchased_list():
    return get_file_response_base64("result")
##############################################################################################################################################################################

# Check Frequency
@app.route("/api/check-frequency", methods=["POST", "GET"])
@authenticate_user_function
def check_frequency():
    return check_frequency_function()
##############################################################################################################################################################################

# Add individual Item to Shopping List
@app.route("/api/addItem/master-nonexpired", methods=["POST"])
@authenticate_user_function
def add_item_master_nonexpired():
    return add_item_to_list("master_nonexpired", "shopping_list")

@app.route("/api/addItem/master-expired", methods=["POST"])
@authenticate_user_function
def add_item_master_expired():
    return add_item_to_list("master_expired", "shopping_list")

@app.route("/api/addItem/purchase-list", methods=["POST"])
@authenticate_user_function
def add_item_result():
    return add_item_to_list("result", "shopping_list")
##############################################################################################################################################################################

# Remove individual Items from the Expired / Non Expired and Shopping List
@app.route("/api/removeItem/master-expired", methods=["POST"])
@authenticate_user_function
@authenticate_user_function
def delete_item_from_master_expired():
    return delete_item_from_list("master_expired")

@app.route("/api/removeItem/shopping-list", methods=["POST"])
@authenticate_user_function
def delete_item_from_shopping_list():
    return delete_item_from_list("shopping_list")

@app.route("/api/removeItem/master-nonexpired", methods=["POST"])
@authenticate_user_function
def delete_item_from_master_nonexpired():
    return delete_item_from_list("master_nonexpired")

@app.route("/api/removeItem/purchase-list", methods=["POST"])
@authenticate_user_function
def delete_item_from_result():
    return delete_item_from_list("result")   
##############################################################################################################################################################################

# Image process upload and compare_image code
@app.route("/api/compare-image", methods=["POST"])
@authenticate_user_function
@authenticate_user_function
def compare_image():
    return compare_image_function()
    
@app.route("/api/image-process-upload", methods=["POST"])
@authenticate_user_function
@authenticate_user_function
def main():
    return main_function()
##############################################################################################################################################################################

# User account setup
@app.route('/api/set-email-create', methods=['POST'])
@authenticate_user_function
def set_email_create():
    return set_email_create_function()
##############################################################################################################################################################################

# User enetered items price 
@app.route('/api/update_price', methods=['POST'])
@authenticate_user_function
def update_price():
    return update_nonexpired_shopping_list_item_price_function()
##############################################################################################################################################################################

# User authentication
@app.route('/api/update_item_name', methods=['POST'])
@authenticate_user_function
def authenticate_user():
    return authenticate_user_function()
##############################################################################################################################################################################

#  Function to move item to food
@app.route('/api/move_to_food', methods=['POST'])
@authenticate_user_function
def move_to_food():
    return move_to_food_function()
##############################################################################################################################################################################

# Preflight requests
@app.route('/api/set-email-create', methods=['OPTIONS'])
def handle_preflight_set_email_create():
    response = jsonify({'status': 'success'})
    response.headers.add("Access-Control-Allow-Origin", "https://my-grocery-home.uc.r.appspot.com")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
    return response

@app.route('/api/image-process-upload', methods=['OPTIONS'])
def handle_preflight_image_process_upload():
    response = jsonify({'status': 'success'})
    response.headers.add("Access-Control-Allow-Origin", "https://my-grocery-home.uc.r.appspot.com")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
    return response

@app.route('/api/compare-image', methods=['OPTIONS'])
def handle_preflight_compare_image():
    response = jsonify({'status': 'success'})
    response.headers.add("Access-Control-Allow-Origin", "https://my-grocery-home.uc.r.appspot.com")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
    return response

@app.route('/api/api/update-master-nonexpired-item-expiry', methods=['OPTIONS'])
def handle_preflight_update_expiry():
    response = jsonify({'status': 'success'})
    response.headers.add("Access-Control-Allow-Origin", "https://my-grocery-home.uc.r.appspot.com")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
    return response
##############################################################################################################################################################################

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))