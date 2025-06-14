from fastapi import APIRouter, Depends, HTTPException
from app.models.request_models import SearchRequest
from app.utils.auth import get_current_user
from app.services.search_service import perform_search
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/search")
async def search_route(request: SearchRequest, user_id: str = Depends(get_current_user)):
    try:
        logger.info(f"🔍 Starting search for image_url: {request.image_url}, top_k: {request.top_k}")
        logger.info(f" User ID: {user_id}")
        results = perform_search(user_id, request.image_url, request.top_k)
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
