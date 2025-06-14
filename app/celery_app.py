from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

UPSTASH_URL = os.getenv("UPSTASH_URL")
if not UPSTASH_URL or not UPSTASH_URL.startswith("rediss://"):
    raise ValueError("Invalid or missing secure Upstash Redis URL.")
# UPSTASH_URL = f"{UPSTASH_URL}?ssl_cert_reqs=optional"
UPSTASH_URL = f"{UPSTASH_URL}?ssl_cert_reqs=CERT_REQUIRED"

# Configure Celery with Redis
celery_app = Celery(
    "textile_search",
    broker=UPSTASH_URL,
    backend=UPSTASH_URL,
    include=["app.tasks"],
    broker_use_ssl={
        'ssl_cert_reqs': None,
        'ssl_ca_certs': None
    },
    redis_backend_use_ssl={
        'ssl_cert_reqs': None,
        'ssl_ca_certs': None
    }
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard timeout
    worker_max_tasks_per_child=200,
    worker_prefetch_multiplier=1,
)
