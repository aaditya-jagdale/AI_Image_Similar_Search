import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from PIL import Image
from app.startup.services import (
    get_upstash_index,
    get_image_embedding,
    is_services_ready
)

image_search_router = APIRouter()

@image_search_router.post("/find_similar")
async def find_similar_images(
    file: UploadFile = File(...), 
    top_k: int = 8
):
    # Check if services are ready
    if not is_services_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image search services are not available. Please check server logs."
        )
    
    index = get_upstash_index()
    if index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upstash Index is not available."
        )

    # Save the uploaded file to a temporary path
    tmp_file_path = None
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must have a filename"
            )
        
        # Validate top_k
        if top_k < 1 or top_k > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_k must be between 1 and 100"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name

        # Load the image as PIL; downstream embedding call handles RGB conversion + preprocessing
        try:
            image = Image.open(tmp_file_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Generate embedding using pre-loaded CLIP model
        vector = get_image_embedding(image)
        
        # Query Upstash vector database
        results = index.query(
            vector=vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append({
                "id": r.id,
                "score": r.score,
                "metadata": r.metadata
            })
        
        return {
            "message": f"Found {len(formatted_results)} similar images.",
            "results": formatted_results
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {str(e)}"
        )
        
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except Exception as e:
                # Log but don't fail if cleanup fails
                print(f"Warning: Failed to remove temporary file {tmp_file_path}: {e}")