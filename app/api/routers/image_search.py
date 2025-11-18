import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.core.config import ONNX_MODEL_PATH, IMG_EMBEDDING_MODEL
from app.services.image_vectorization_service import ImageVectorizerONNX
from app.services.simple_vector_db import SimpleVectorDB
from app.services.supabase_service import SupabaseService
image_search_router = APIRouter()

# Same as in ingest.py, get these from your config
MODEL_ID = IMG_EMBEDDING_MODEL

# Initialize services on startup
try:
    vectorizer = ImageVectorizerONNX(onnx_path=ONNX_MODEL_PATH, model_id=MODEL_ID)
    db_service = SimpleVectorDB()
    supabase_service = SupabaseService()
except Exception as e:
    print(f"CRITICAL: Failed to initialize services on startup: {e}")
    vectorizer = None
    db_service = None
    supabase_service = None
@image_search_router.post("/find_similar")
async def find_similar_images(
    file: UploadFile = File(...), 
    top_k: int = 8
):
    """
    Receives an image, generates an embedding, and returns similar items.
    """
    if not vectorizer or not db_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Core services are not available."
        )

    # Your ImageVectorizerONNX takes a file path, not bytes.
    # So, we must save the uploaded file to a temporary path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name

    try:
        # 1. Generate Embedding
        query_embedding = vectorizer.encode_single(tmp_file_path)
        
        # 2. Query Database
        results = db_service.query_image(
            query_embedding=query_embedding, 
            top_k=top_k
        )

        all_ids = results.get("ids", [[]])[0]

        get_products = supabase_service.get_multiple_items_by_ids(all_ids)
        print(f"Retrieved {len(get_products)} products from Supabase for IDs: {all_ids}")
        
        return {
            "message": f"Found {len(results.get('ids', [[]])[0])} similar images.",
            # "products": get_products,
            "results": results
        }

    except Exception as e:
        print(f"Error during image search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {e}"
        )
    finally:
        # 3. Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)