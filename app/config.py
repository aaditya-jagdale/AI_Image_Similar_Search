import os

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-large-patch14"
# IMAGE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset')) # Removed
INDEX_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'product_index.faiss'))
IDS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'product_ids.txt'))
TOP_K = 5

# Environment variable for KMeans compatibility
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Flask specific configs (optional)
# Sets DEBUG mode based on FLASK_DEBUG env var, defaulting to False (production-safe)
DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 't')
HOST = '0.0.0.0'
PORT = 5001