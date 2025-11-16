import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = "gemini-flash-lite-latest"
IMG_EMBEDDING_MODEL = "Qdrant/clip-ViT-B-32-vision"
ONNX_MODEL_PATH = "clip_vision_static.onnx"
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_NAME="dataset_collection"
CHROMA_DIR="dataset_db"