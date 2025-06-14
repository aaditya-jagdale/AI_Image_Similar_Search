from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Header
from app.search import validate_token
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/celery_task_status/{task_id}")
async def celery_task_status(task_id: str, authorization: str = Header(...)):
    """
    Get the status of a Celery background task by task_id. Requires JWT in Authorization header.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    user_id = validate_token(token)
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    logger.info(f"Checking Celery status for task: {task_id} by user: {user_id}")
    result = AsyncResult(task_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Task not found in Celery backend")
    
    return {
        "task_id": task_id,
        "state": result.state,
        "info": result.info if hasattr(result, 'info') else None,
        "result": result.result if hasattr(result, 'result') else None
    }
