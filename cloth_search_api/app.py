from flask import Flask, request, jsonify
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import json
import os
import faiss
import traceback
from dotenv import load_dotenv
from download_faiss import download_faiss_files


app = Flask(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------- Setup Environment ---------------------
load_dotenv() # Load environment variables from .env file
PASSKEY = os.getenv("PASSKEY")

# ------------------------- Setup Paths ---------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_index_path = os.path.join(script_dir, "faiss/search.faiss")
metadata_path = os.path.join(script_dir, "metadata/search_metadata.json")

# ------------------------ Load Model -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

try:
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    print("[INFO] CLIP model and processor loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load CLIP model: {e}")
    raise

# ------------------------ Load Metadata --------------------------

try:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"[INFO] Loaded metadata with {len(metadata)} entries.")
except Exception as e:
    print(f"[ERROR] Could not load metadata: {e}")
    metadata = []

# ------------------------ Load FAISS Index -----------------------

try:
    index = faiss.read_index(faiss_index_path)
    print(f"[INFO] FAISS index loaded. Dimension: {index.d}, Total items: {index.ntotal}")
except Exception as e:
    print(f"[ERROR] Could not load FAISS index: {e}")
    index = None

# ------------------------ Feature Extractor ----------------------

def extract_features_clip(image: Image.Image) -> np.ndarray:
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        vector = image_features.cpu().numpy().astype("float32")
        print(f"[DEBUG] Extracted feature vector shape: {vector.shape}")
        return vector
    except Exception as e:
        print(f"[ERROR] Failed to extract features: {e}")
        raise

# ------------------------ Routes ---------------------------------

@app.route("/search", methods=["POST"])
def search():
    if index is None:
        return jsonify({"error": "FAISS index not loaded"}), 500

    # --- Passkey Check ---
    request_passkey = request.form.get("passkey")
    if not request_passkey or request_passkey != PASSKEY:
        return jsonify({"error": "Unauthorized - Invalid or missing passkey"}), 401
    # --- End Passkey Check ---

    try:
        if "image" not in request.files:
            return jsonify({"error": "Missing image file"}), 400

        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")
        print("[INFO] Received image and converted to RGB.")

        query_vector = extract_features_clip(image)
        print(f"[INFO] Query vector shape: {query_vector.shape}, Index dim: {index.d}")

        if query_vector.shape[1] != index.d:
            return jsonify({
                "error": "Embedding dimension mismatch",
                "query_dim": query_vector.shape[1],
                "index_dim": index.d
            }), 500

        k = int(request.form.get("k", 5))
        print(f"[INFO] Performing search with top-{k} results.")
        distances, indices = index.search(query_vector, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(metadata):
                results.append({
                    "distance": float(dist),
                    "result": metadata[idx]
                })
            else:
                print(f"[WARNING] Index {idx} out of bounds in metadata.")

        print(f"[INFO] Returning {len(results)} search results.")
        return jsonify(results)

    except Exception as e:
        print("[ERROR] Exception occurred in /search endpoint.")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/download_faiss", methods=["GET"])
def download_faiss():
    download_faiss_files()
    return jsonify({"status": "ok"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ------------------------ Entrypoint -----------------------------

if __name__ == "__main__":
    print("[INFO] Starting server on 0.0.0.0:8000")
    # app.run(host="0.0.0.0", port=8000, debug=True)
    app.run(host="0.0.0.0", port=8000)
