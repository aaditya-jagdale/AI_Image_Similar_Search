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
DEBUG = True # Set to False in production
HOST = '0.0.0.0'
PORT = 5001 