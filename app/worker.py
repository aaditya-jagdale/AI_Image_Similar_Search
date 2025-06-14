from app.celery_app import celery_app
import logging
from app.tasks import process_embeddings  # Explicitly import the task

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting Celery worker...")
    # Register the task explicitly
    celery_app.tasks.register(process_embeddings)
    celery_app.worker_main(['worker', '--loglevel=INFO']) 