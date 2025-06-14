from fastapi import APIRouter, Header, HTTPException, Depends
from app.utils.auth import get_current_user
from app.services.upload_service import start_upload_embeddings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload_embeddings")
async def upload_embeddings_route(authorization: str = Header(...)):
    try:
        return await start_upload_embeddings(authorization)
    except Exception as e:
        logger.error(f"Error initiating upload_embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initiate upload process: {e}")
