from flask import Blueprint

import modules.core as core
import modules.inventory_handlers as handlers

bp = Blueprint('inventory', __name__, url_prefix='/api')

@bp.route('/deleteAll/master-nonexpired', methods=['POST'])
@core.authenticate_user_function
def delete_all_master_nonexpired():
    return handlers.delete_all_items('master_nonexpired')

@bp.route('/deleteAll/master-expired', methods=['POST'])
@core.authenticate_user_function
def delete_all_master_expired():
    return handlers.delete_all_items('master_expired')

@bp.route('/deleteAll/shopping-list', methods=['POST'])
@core.authenticate_user_function
def delete_all_shopping():
    return handlers.delete_all_items('shopping_list')

@bp.route('/deleteAll/purchase-list', methods=['POST'])
@core.authenticate_user_function
def delete_all_purchase():
    return handlers.delete_all_items('result')

@bp.route('/add-custom-item', methods=['POST'])
@core.authenticate_user_function
def add_custom_item():
    return handlers.add_custom_item()

@bp.route('/update-master-nonexpired-item-expiry', methods=['POST'])
@core.authenticate_user_function
def update_master_nonexpired_item_expiry():
    return handlers.update_master_nonexpired_item_expiry()

@bp.route('/get-master-expired-list', methods=['GET'])
@core.authenticate_user_function
def get_master_expired():
    return handlers.get_file_response_base64('master_expired')

@bp.route('/get-shopping-list', methods=['GET'])
@core.authenticate_user_function
def get_shopping_list():
    return handlers.get_file_response_base64('shopping_list')

@bp.route('/get-master-nonexpired-list', methods=['GET'])
@core.authenticate_user_function
def get_master_nonexpired():
    return handlers.get_file_response_base64('master_nonexpired')

@bp.route('/get-purchased-list', methods=['GET'])
@core.authenticate_user_function
def get_purchased_list():
    return handlers.get_file_response_base64('result')

@bp.route('/check-frequency', methods=['POST', 'GET'])
@core.authenticate_user_function
def check_frequency():
    return handlers.check_frequency()

@bp.route('/addItem/master-nonexpired', methods=['POST'])
@core.authenticate_user_function
def add_item_master_nonexpired():
    return handlers.add_item_to_list('master_nonexpired', 'shopping_list')

@bp.route('/addItem/master-expired', methods=['POST'])
@core.authenticate_user_function
def add_item_master_expired():
    return handlers.add_item_to_list('master_expired', 'shopping_list')

@bp.route('/addItem/purchase-list', methods=['POST'])
@core.authenticate_user_function
def add_item_result():
    return handlers.add_item_to_list('result', 'shopping_list')

@bp.route('/removeItem/master-expired', methods=['POST'])
@core.authenticate_user_function
def delete_item_from_master_expired():
    return handlers.delete_item_from_list('master_expired')

@bp.route('/removeItem/shopping-list', methods=['POST'])
@core.authenticate_user_function
def delete_item_from_shopping_list():
    return handlers.delete_item_from_list('shopping_list')

@bp.route('/removeItem/master-nonexpired', methods=['POST'])
@core.authenticate_user_function
def delete_item_from_master_nonexpired():
    return handlers.delete_item_from_list('master_nonexpired')

@bp.route('/removeItem/purchase-list', methods=['POST'])
@core.authenticate_user_function
def delete_item_from_result():
    return handlers.delete_item_from_list('result')
