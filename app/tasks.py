from .celery_app import celery_app
from .chroma_provider import upsert_to_chroma
from .supabase_client import SupabaseClient
from .task_tracker import task_tracker
import logging
from typing import Dict, List
import time

logger = logging.getLogger(__name__)

@celery_app.task(name='app.tasks.process_embeddings', bind=True)
def process_embeddings(self, user_id: str, access_token: str) -> Dict:
    """
    Background task to process embeddings for a user's items
    """
    task_id = self.request.id
    logger.info(f"Starting embedding process for task {task_id}")
    task_tracker.add_task(task_id, user_id)
    start_time = time.time()
    try:
        # Initialize Supabase client
        task_tracker.update_progress(task_id, 0, 0, "Connecting to Supabase")
        supabase_client = SupabaseClient(access_token)
        
        # Fetch items
        task_tracker.update_progress(task_id, 0, 0, "Fetching items from database")
        items = supabase_client.get_all_items()
        
        # Update total items count
        total_items = len(items)
        task_tracker.update_progress(task_id, 0, total_items, "Preparing items for processing")
        
        # Convert items to the format expected by upsert_to_chroma
        data = [item.model_dump() for item in items]
        
        # Process and upload the data using ChromaDB
        task_tracker.update_progress(task_id, 0, total_items, "Starting ChromaDB processing")
        response = upsert_to_chroma(
            user_id=user_id,
            data=data,
            persist_directory="./chroma",
            task_id=task_id
        )
        
        time_taken = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        result = {
            "status": "completed",
            "timeTaken": time_taken,
            "message": "Upload completed successfully",
            "failedItems": response["failed"],
            "totalItems": len(data),
            "failedItemsList": response["failures"],
            "items": data,
            "upserted": response.get("upserted", 0)
        }
        
        task_tracker.complete_task(task_id, result)
        return result
        
    except Exception as e:
        error_msg = f"Error during background processing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        task_tracker.fail_task(task_id, error_msg)
        return {
            "status": "failed",
            "error": str(e)
        }