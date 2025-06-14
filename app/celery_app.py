from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Celery
celery_app = Celery(
    'textile_search',
    broker=os.getenv('UPSTASH_URL'),
    backend=os.getenv('UPSTASH_URL'),
    include=['app.tasks']  # This tells Celery where to find our tasks
)

# Optional configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,  # Enable task tracking
    task_time_limit=3600,    # 1 hour timeout
    worker_max_tasks_per_child=200,  # Restart worker after 200 tasks
    worker_prefetch_multiplier=1  # Process one task at a time
) 