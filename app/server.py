from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from PIL import Image
import io
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import os
from dotenv import load_dotenv
from .search import search_similar, processor, model
load_dotenv()

if os.getenv("QDRANT_URL") is None or os.getenv("QDRANT_API_KEY") is None:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

# Connect to Qdrant Cloud
print("[INFO] Connecting to Qdrant Cloud...")
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
collection_name = "textile_designs"

print(f"[INFO] Collection '{collection_name}' connected.")
# Define FastAPI app
app = FastAPI(title="Textile Design Vector Search API")

# Helper to extract image embedding
def get_image_embedding(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings[0].numpy().tolist()

@app.post("/search")
def search(image_url: str, top_k: int = 10):
    results = search_similar(image_url, top_k)
    return results
