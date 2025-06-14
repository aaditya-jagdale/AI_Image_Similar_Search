from fastapi import APIRouter, Header, HTTPException, Depends
from app.utils.auth import get_current_user
from app.services.task_service import get_status
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/task_status/{task_id}")
async def task_status_route(task_id: str, authorization: str = Header(...)):
    try:
        return get_status(task_id, authorization)
    except Exception as e:
        logger.error(f"Error getting task status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {e}")
