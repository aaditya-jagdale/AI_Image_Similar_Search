from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from pydantic import BaseModel
from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams # Removed unused import
from PIL import Image
import io
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import os
from dotenv import load_dotenv
from .search import search_similar, processor, model, validate_token
from typing import Optional
import jwt
from supabase import create_client, Client
from .embed_and_upload import process_and_upload_data
load_dotenv()
import logging
import uvicorn # Added
import time
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.getenv("QDRANT_URL") is None or os.getenv("QDRANT_API_KEY") is None or os.getenv("SUPABASE_URL") is None or os.getenv("SUPABASE_KEY") is None:
    logger.error("QDRANT_URL, QDRANT_API_KEY, SUPABASE_URL, and SUPABASE_KEY must be set in environment variables.")
    raise ValueError("QDRANT_URL, QDRANT_API_KEY, SUPABASE_URL, and SUPABASE_KEY must be set")

# Initialize Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Connect to Qdrant Cloud
logger.info("[INFO] Connecting to Qdrant Cloud...")
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
collection_name = "textile_designs"
 
logger.info(f"[INFO] Collection '{collection_name}' connected.") # Changed to logger
# Define FastAPI app
app = FastAPI(title="Textile Design Vector Search API")

# Define request body model for the search endpoint
class SearchRequest(BaseModel):
    image_url: str
    top_k: int = 10

class TextileItem(BaseModel):
    design_no: str
    width: str
    stock: str
    GSM: str
    image: str
    source_pdf: str

class UploadRequest(BaseModel):
    items: list[TextileItem]

async def get_current_user(authorization: str = Header(...)) -> str:
    """Validate the authorization token and return the user ID."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    user_id = validate_token(token)
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return user_id

print("--------------------------------\n\n\n")
print("👇🏻 Starting server...👇🏻")
print("\n\n\n--------------------------------")

@app.post("/search")
async def search(request: SearchRequest, user_id: str = Depends(get_current_user)):
    try:
        logger.info(f"🔍 Starting search for image_url: {request.image_url}, top_k: {request.top_k}")
        logger.info(f" User ID: {user_id}")
        results = await search_similar(request.image_url, qdrant_client, user_id, request.top_k)
        logger.info("✅ Search completed")
        if results is None:
            logger.warning("Search returned no results or an error occurred upstream.")
            return {"status": "ok", "results": []}
    except HTTPException as e:
        logger.error(f"HTTPException during search: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Qdrant query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed due to an internal error: {e}")

    return {"status": "ok", "results": results}

@app.post("/upload_embeddings")
async def upload_embeddings(request: UploadRequest, user_id: str = Depends(get_current_user)):
    try:
        logger.info(f"🔍 Starting upload_embeddings for user_id: {user_id}")
        logger.info(f"Processing {len(request.items)} items")
        start_time = time.time()
        # Convert items to the format expected by process_and_upload_data
        data = [item.model_dump() for item in request.items]
        
        # Process and upload the data
        response = process_and_upload_data(data, user_id)
        logger.info("✅ Upload completed")
    except Exception as e:
        logger.error(f"Error during upload_embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed due to an internal error: {e}")
    return {
        "status": "ok",
        "timeTaken": time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
        "message": response["message"],
        "totalItems": response["total_items"],
        "failedItems": response["failed_items"],
        "failedItemsList": response["failed_items_list"]
    }

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "Never gonna give you up"}

if __name__ == "__main__": # Added main block
    port = int(os.getenv("PORT", 8000)) # Railway provides PORT env var
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
