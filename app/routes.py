import os
import traceback
import base64
import io
import requests  # Add requests import
from flask import Blueprint, request, jsonify, current_app
from PIL import Image, UnidentifiedImageError # Add PIL import and specific error
from .services.handle_build_index import handle_build_index
from .services.handle_search import handle_image_search # Import the new handler
from .services.handle_build_index import handle_build_index_from_json # Import the new JSON index builder
from .services.download_faiss import download_faiss_files # Import the download function

api_bp = Blueprint('api', __name__, url_prefix='/api/v1') # Added version prefix

@api_bp.route('/')
def home():
    return jsonify({"message": "The server is running"})

@api_bp.route('/build_index', methods=['POST'])
def build_index_endpoint():
    if 'pdf_files' not in request.files:
        return jsonify({"error": "No 'pdf_files' field in the request"}), 400

    pdf_files = request.files.getlist('pdf_files')

    if not pdf_files or all(f.filename == '' for f in pdf_files):
        return jsonify({"error": "No PDF files selected or uploaded"}), 400

    # Optional: Add check for file types (e.g., ensure they are PDFs)
    for f in pdf_files:
        if not f.filename.lower().endswith('.pdf'):
            return jsonify({"error": f"File '{f.filename}' is not a PDF."}), 400

    try:
        # Call the service function with the list of FileStorage objects
        response, status_code = handle_build_index(pdf_files)
        return response, status_code
    except Exception as e:
        # Log the unexpected error
        current_app.logger.error(f"Unexpected error during index build from PDFs: {e}")
        traceback.print_exc() # Print traceback for detailed debugging
        return jsonify({"error": "An internal server error occurred during index building."}), 500

# --- New route to build index from JSON file ---
@api_bp.route('/build_index_from_json', methods=['POST'])
def build_index_from_json_endpoint():
    """Endpoint to trigger building the FAISS index from output.json."""
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No JSON data provided in the request body"}), 400

    try:
        # Directly call the service function, assuming output.json is in the root
        # If you need to specify the path via request, modify here
        response, status_code = handle_build_index_from_json(json_data)
        return response, status_code
    except Exception as e:
        # Log unexpected errors during the process
        current_app.logger.error(f"Unexpected error during index build from JSON: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred during JSON index building."}), 500

@api_bp.route('/download_faiss', methods=['POST'])
def download_faiss_endpoint():
    """Endpoint to download FAISS index files from Supabase storage."""
    try:
        success, message = download_faiss_files()
        if success:
            return jsonify({
                "message": message,
                "status": "success"
            }), 200
        else:
            return jsonify({
                "error": message,
                "status": "error"
            }), 500
    except Exception as e:
        current_app.logger.error(f"Unexpected error during FAISS download: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "An internal server error occurred during FAISS download.",
            "status": "error"
        }), 500

@api_bp.app_errorhandler(404)
def handle_404(err):
    return jsonify(error=f"API endpoint not found: {request.path}"), 404

# --- New Image Search Route --- 
@api_bp.route('/search/image', methods=['POST'])
def search_image_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Optional: Add further checks for allowed file types/extensions
    
    # Get optional 'k' parameter for number of results
    k = request.form.get('k', default=10, type=int)

    return handle_image_search(image_file, k=k)