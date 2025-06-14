from app.search import validate_token
from app.tasks import process_embeddings
import logging

async def start_upload_embeddings(authorization: str):
    if not authorization.startswith("Bearer "):
        raise Exception("Invalid authorization header")
    token = authorization.split(" ")[1]
    user_id = validate_token(token)
    if not user_id:
        raise Exception("Invalid token")
    logging.info(f"Starting upload process for user: {user_id}")
    task = process_embeddings.delay(user_id, token)
    logging.info(f"Created Celery task with ID: {task.id}")
    return {
        "status": "processing",
        "message": "Upload process started in background",
        "task_id": task.id
    }
