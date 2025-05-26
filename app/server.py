from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams # Removed unused import
from PIL import Image
import io
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import os
from dotenv import load_dotenv
from .search import search_similar, processor, model
load_dotenv()
import logging
import uvicorn # Added

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if os.getenv("QDRANT_URL") is None or os.getenv("QDRANT_API_KEY") is None:
    logger.error("QDRANT_URL and QDRANT_API_KEY must be set in environment variables.") # Changed to logger
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

# Connect to Qdrant Cloud
logger.info("[INFO] Connecting to Qdrant Cloud...") # Changed to logger
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
collection_name = "textile_designs"
 
logger.info(f"[INFO] Collection '{collection_name}' connected.") # Changed to logger
# Define FastAPI app
app = FastAPI(title="Textile Design Vector Search API")
print("--------------------------------\n\n\n")
print("👇🏻 Starting server...👇🏻")
print("\n\n\n--------------------------------")
# Helper to extract image embedding (Not used in server.py, consider moving to search.py or removing if not needed elsewhere)
# def get_image_embedding(image_bytes: bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         embeddings = model.get_image_features(**inputs)
#     return embeddings[0].numpy().tolist()

@app.post("/search")
async def search(image_url: str, top_k: int = 10):
    try:
        logger.info(f"🔍 Starting search for image_url: {image_url}, top_k: {top_k}") # Changed print to logger
        results = await search_similar(image_url, top_k)
        logger.info("✅ Search completed") # Changed print to logger
        if results is None: # Added check for None results
            logger.warning("Search returned no results or an error occurred upstream.")
            # Return empty list or appropriate HTTP response if search_similar can return None on error
            return {"status": "ok", "results": []} 
    except HTTPException as e:
        logger.error(f"HTTPException during search: {e.detail}")
        raise # Re-raise HTTPException
    except Exception as e:
        logger.error(f"Qdrant query failed: {e}", exc_info=True) # Added exc_info for better debugging
        raise HTTPException(status_code=500, detail=f"Search failed due to an internal error: {e}")

    return {"status": "ok", "results": results}

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "Never gonna give you up"}

if __name__ == "__main__": # Added main block
    port = int(os.getenv("PORT", 8000)) # Railway provides PORT env var
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
