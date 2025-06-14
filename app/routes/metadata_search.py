from fastapi import APIRouter, Depends, HTTPException
from app.models.request_models import MetadataSearchRequest
from app.utils.auth import get_current_user
from app.services.metadata_service import perform_metadata_search
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/metadata_search")
async def metadata_search_route(
    design_no: str = "",
    width: str = "",
    stock: str = "",
    GSM: str = "",
    source_pdf: str = "",
    top_k: int = 10,
    user_id: str = Depends(get_current_user)
):
    try:
        results = await perform_metadata_search(
            design_no, width, stock, GSM, source_pdf, user_id, top_k
        )
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
