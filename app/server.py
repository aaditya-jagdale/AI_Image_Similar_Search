from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import os
from dotenv import load_dotenv
from app.search import validate_token
from typing import Optional, List, Dict
import jwt
from supabase import create_client, Client
import logging
import uvicorn
import time
import chromadb
from app.chroma_provider import upsert_to_chroma, search_images
from app.text_search import metadata_search, MetadataSearchRequest
from app.supabase_client import SupabaseClient, TextileProduct
# Configure logging
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.getenv("SUPABASE_URL") is None or os.getenv("SUPABASE_KEY") is None:
    logger.error("SUPABASE_URL, and SUPABASE_KEY must be set in environment variables.")
    raise ValueError("SUPABASE_URL, and SUPABASE_KEY must be set")

# Initialize Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Initialize ChromaDB client
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma")
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)

# Define FastAPI app
app = FastAPI(title="Textile Design Vector Search API")

# Add CORS middleware
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    os.getenv("VITE_URL"),
]

# allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body model for the search endpoint
class SearchRequest(BaseModel):
    image_url: str
    top_k: int = 10


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
        
        # Use ChromaDB search_images function
        results = search_images(
            user_id=user_id,
            image_path_or_url=request.image_url,
            top_k=request.top_k
        )
        
        logger.info("✅ Search completed")
        if not results:
            logger.warning("Search returned no results or an error occurred upstream.")
            return {"status": "ok", "results": []}
            
    except HTTPException as e:
        logger.error(f"HTTPException during search: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed due to an internal error: {e}")

    return {"status": "ok", "results": results}

@app.get("/metadata_search")
async def metadata_search_endpoint(
    design_no: Optional[str] = None,
    width: Optional[str] = None,
    stock: Optional[str] = None,
    GSM: Optional[str] = None,
    source_pdf: Optional[str] = None,
    top_k: int = 10,
    user_id: str = Depends(get_current_user)
):
    """Endpoint to perform a metadata-based search on the user's collection."""
    try:
        # Create search request from query parameters
        search_request = MetadataSearchRequest(
            design_no=design_no or "",
            width=width or "",
            stock=stock or "",
            GSM=GSM or "",
            source_pdf=source_pdf or ""
        )
        
        logger.info(f"🔍 Starting metadata search for query: '{search_request}', top_k: {top_k}, user_id: {user_id}")
        results = await metadata_search(search_request, chroma_client, user_id, top_k)

        if results is None:
            logger.warning("Metadata search returned no results or an upstream error occurred")
            return {"status": "ok", "results": []}

        logger.info("✅ Metadata search completed")
    except HTTPException as e:
        logger.error(f"HTTPException during metadata search: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Metadata search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Metadata search failed due to an internal error: {e}")

    return {"status": "ok", "results": results}

@app.post("/upload_embeddings")
async def upload_embeddings(user_id: str = Depends(get_current_user)):
    try:
        start_time = time.time()
        supabase_client = SupabaseClient(user_id)
        items = supabase_client.get_all_items()
        logger.info(f"🔍 Starting upload_embeddings for user_id: {user_id}")
        logger.info(f"Processing {len(items)} items")
        
        # Convert items to the format expected by upsert_to_chroma
        data = [item.model_dump() for item in items]
        
        # Process and upload the data using ChromaDB
        response = upsert_to_chroma(
            user_id=user_id,
            data=data,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        
        logger.info("✅ Upload completed")
    except Exception as e:
        logger.error(f"Error during upload_embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed due to an internal error: {e}")
    
    return {
        "status": "ok",
        "timeTaken": time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
        "message": "Upload completed successfully",
        "totalItems": len(data),
        "failedItems": response["failed"],
        "failedItemsList": response["failures"]
    }

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "Never gonna give you up"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
