from flask import Blueprint

import modules.core as core
import modules.image_handlers as handlers

bp = Blueprint('image', __name__, url_prefix='/api')

@bp.route('/compare-image', methods=['POST'])
@core.authenticate_user_function
def compare_image():
    return handlers.compare_image_function()

@bp.route('/image-process-upload', methods=['POST'])
@core.authenticate_user_function
def image_process_upload():
    return handlers.main_function()
