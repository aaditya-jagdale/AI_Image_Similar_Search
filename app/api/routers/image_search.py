import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from PIL import Image
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
            detail=f"Core services are not available.\n vectorizer: {vectorizer is not None}, db_service: {db_service is not None}"
        )

    # Your ImageVectorizerONNX takes a file path, not bytes.
    # So, we must save the uploaded file to a temporary path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name

    try:
        # 1. Load the image
        base_img = Image.open(tmp_file_path)
        base_img = base_img.convert("RGB")
        
        # 2. Create 4 transformed images
        # - Rotate 90 degrees
        img_90 = base_img.rotate(90, expand=True)
        # - Rotate 180 degrees
        img_180 = base_img.rotate(180, expand=True)
        # - Rotate 135 degrees
        img_135 = base_img.rotate(135, expand=True)
        # - Zoom in by 25% (scale by 1.25x)
        width, height = base_img.size
        img_zoom = base_img.resize((int(width * 1.25), int(height * 1.25)), Image.Resampling.LANCZOS)
        
        # 3. Encode all 4 transformed images
        embeddings = [
            vectorizer.encode_single(img_90),
            vectorizer.encode_single(img_180),
            vectorizer.encode_single(img_135),
            vectorizer.encode_single(img_zoom)
        ]
        
        # 4. Query the database with all 4 encoded images
        all_results = []
        for embedding in embeddings:
            query_results = db_service.query_image(
                query_embedding=embedding,
                top_k=top_k * 2  # Get more results to have enough after deduplication
            )
            all_results.append(query_results)
        
        # 5. Combine all results into a single list
        combined_ids = []
        combined_distances = []
        combined_metadatas = []
        
        for result in all_results:
            ids = result.get("ids", [[]])[0] if result.get("ids") else []
            distances = result.get("distances", [[]])[0] if result.get("distances") else []
            metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
            
            for i, id_val in enumerate(ids):
                combined_ids.append(id_val)
                combined_distances.append(distances[i] if i < len(distances) else None)
                combined_metadatas.append(metadatas[i] if i < len(metadatas) else None)
        
        # 6. Clean the list - remove duplicates by ID, keeping the one with lowest distance
        unique_results = {}
        for i, id_val in enumerate(combined_ids):
            current_distance = combined_distances[i]
            current_metadata = combined_metadatas[i]
            
            if id_val not in unique_results:
                unique_results[id_val] = {
                    "id": id_val,
                    "distance": current_distance,
                    "metadata": current_metadata
                }
            else:
                # Keep the result with the lower distance
                existing_distance = unique_results[id_val]["distance"]
                
                # If current has valid distance and existing doesn't, use current
                if current_distance is not None and existing_distance is None:
                    unique_results[id_val] = {
                        "id": id_val,
                        "distance": current_distance,
                        "metadata": current_metadata
                    }
                # If both have valid distances, keep the one with lower distance
                elif current_distance is not None and existing_distance is not None:
                    if current_distance < existing_distance:
                        unique_results[id_val] = {
                            "id": id_val,
                            "distance": current_distance,
                            "metadata": current_metadata
                        }
        
        # 7. Sort by distance and get top 8
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x["distance"] if x["distance"] is not None else float('inf')
        )[:top_k]
        
        # Format results to match the original structure
        final_ids = [r["id"] for r in sorted_results]
        final_distances = [r["distance"] for r in sorted_results]
        final_metadatas = [r["metadata"] for r in sorted_results]
        
        results = {
            "ids": [final_ids],
            "distances": [final_distances],
            "metadatas": [final_metadatas]
        }

        all_ids = final_ids

        get_products = supabase_service.get_multiple_items_by_ids(all_ids)
        print(f"Retrieved {len(get_products)} products from Supabase for IDs: {all_ids}")
        
        return {
            "message": f"Found {len(final_ids)} similar images.",
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