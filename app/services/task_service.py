from app.search import validate_token
from app.task_tracker import task_tracker
from fastapi import HTTPException
import logging
import datetime

def get_status(task_id: str, authorization: str):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    user_id = validate_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    logging.info(f"Checking status for task: {task_id} by user: {user_id}")
    task_status = task_tracker.get_task_status(task_id)
    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    if task_status.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this task")
    processed = task_status.get("processed_items", 0)
    total = task_status.get("total_items", 0)
    start_time = task_status.get("start_time")
    end_time = task_status.get("end_time")
    time_taken = None
    est_time_left = None
    if start_time:
        if end_time:
            time_taken = str(end_time - start_time)
        else:
            now = datetime.datetime.now()
            time_taken = str(now - start_time)
            if processed > 0 and total > 0:
                elapsed = (now - start_time).total_seconds()
                est_time_left = elapsed * (total - processed) / processed if processed else None
                if est_time_left:
                    est_time_left = str(datetime.timedelta(seconds=int(est_time_left)))
    return {
        "task_id": task_id,
        "status": task_status.get("status"),
        "progress": task_status.get("progress"),
        "processed_items": processed,
        "total_items": total,
        "remaining_items": task_status.get("remaining_items"),
        "current_operation": task_status.get("current_operation"),
        "completed_batches": task_status.get("result", {}).get("completed_batches"),
        "total_batches": task_status.get("result", {}).get("total_batches"),
        "error": task_status.get("error"),
        "last_updated": task_status.get("last_updated"),
        "time_taken": time_taken,
        "estimated_time_left": est_time_left
    }
