import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Environment / API config
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = "gemini-flash-lite-latest"
IMG_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

_ROOT = Path(__file__).resolve().parents[2]
_default_data_model = _ROOT / "data" / "clip_vision_static.onnx"
if _default_data_model.exists():
	ONNX_MODEL_PATH = str(_default_data_model)
else:
	# Allow overriding via env var; if not set, keep the original relative name
	ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "clip_vision_static.onnx")

SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_NAME = "dataset_collection"
CHROMA_DIR = "dataset_db_multi_angle"