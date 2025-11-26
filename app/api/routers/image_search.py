import os
import shutil
import tempfile
from collections import defaultdict
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
        
        # Rotations for multi-query approach: 0, 35, and 90 degrees
        rotation_angles = [0, 35, 90]
        
        # Dictionary to track results: {id: {'count': int, 'scores': [float], 'metadata': dict}}
        result_frequency = defaultdict(lambda: {'count': 0, 'scores': [], 'metadata': None})
        
        # Query 3 times with different rotations
        for angle in rotation_angles:
            # Rotate the image
            if angle == 0:
                rotated_image = image
            else:
                rotated_image = image.rotate(angle, expand=True)
            
            # Generate embedding for rotated image
            vector = get_image_embedding(rotated_image)
            
            # Query Upstash vector database
            query_results = index.query(
                vector=vector.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            # Track results by frequency
            for r in query_results:
                result_frequency[r.id]['count'] += 1
                result_frequency[r.id]['scores'].append(r.score)
                # Store metadata (will be overwritten but should be same for same id)
                if result_frequency[r.id]['metadata'] is None:
                    result_frequency[r.id]['metadata'] = r.metadata
        
        # Sort results by frequency (descending), then by average score (descending)
        sorted_results = []
        for result_id, data in result_frequency.items():
            avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0.0
            sorted_results.append({
                'id': result_id,
                'frequency': data['count'],
                'avg_score': avg_score,
                'metadata': data['metadata']
            })
        
        # Sort by frequency (descending), then by average score (descending)
        sorted_results.sort(key=lambda x: (x['frequency'], x['avg_score']), reverse=True)
        
        # Take top_k results
        formatted_results = sorted_results[:top_k]
        
        # Format final results (remove frequency and avg_score from response, keep for internal use)
        final_results = []
        for r in formatted_results:
            final_results.append({
                "id": r['id'],
                "score": r['avg_score'],
                "frequency": r['frequency'],
                "metadata": r['metadata']
            })

        output = {
            "message": f"Found {len(final_results)} similar images using multi-query approach.",
            "results": final_results
        }

        print(output)
        
        return output

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