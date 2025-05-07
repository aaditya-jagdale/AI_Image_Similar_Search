import os
import traceback
from flask import Flask, jsonify, request


from .services.services import ClipService, FaissIndexService
from .routes import api_bp
from . import config 

def create_app():
    """Application Factory: Creates and configures the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config)
    clip_service = ClipService(model_name=app.config['MODEL_NAME'])

    try:
        clip_service.load() 
    except RuntimeError as e:
        print(f"Error during service initialization: {e}")
        print("API might not function correctly until models/index are loaded/built.")

    
    
    app.extensions['clip_service'] = clip_service
    app.register_blueprint(api_bp)

    @app.errorhandler(404)
    def handle_404_global(err):
        return jsonify(error=f"Resource not found: {request.path}"), 404

    @app.errorhandler(500)
    def handle_500_global(err):
        original_exception = getattr(err, "original_exception", err)
        print(f"Internal Server Error: {original_exception}\n{traceback.format_exc()}")
        return jsonify(error="An internal server error occurred."), 500

    @app.errorhandler(Exception)
    def handle_uncaught_exception(err):
        print(f"Unhandled Exception: {err}\n{traceback.format_exc()}")
        error_message = "An unexpected error occurred." if not app.config['DEBUG'] else str(err)
        return jsonify(error=error_message), 500

    print("Flask app created and configured.")
    
    @app.route('/')
    def root_home(): # noqa
        return jsonify({"message": "Welcome! API available at /api/v1/"})

    return app