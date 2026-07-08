from flask import Blueprint

import modules.core as core
import modules.user_handlers as handlers

bp = Blueprint('user', __name__, url_prefix='/api')

@bp.route('/create-missing-chatgpt-files', methods=['POST'])
@core.authenticate_user_function
def create_missing_chatgpt_files():
    return handlers.create_missing_chatgpt_files_function()

@bp.route('/check-user-files', methods=['GET'])
@core.authenticate_user_function
def check_user_files():
    return handlers.check_user_files_function()

@bp.route('/initialize-user-complete', methods=['POST'])
@core.authenticate_user_function
def initialize_user_complete():
    return handlers.initialize_user_complete_function()

@bp.route('/cleanup-user-files', methods=['POST'])
@core.authenticate_user_function
def cleanup_user_files():
    return handlers.cleanup_user_files_function()

@bp.route('/set-email-create', methods=['POST'])
@core.authenticate_user_function
def set_email_create():
    return handlers.set_email_create_function()

@bp.route('/update_price', methods=['POST'])
@core.authenticate_user_function
def update_price():
    return handlers.update_purchased_nonexpired_shopping_item_price_function()

