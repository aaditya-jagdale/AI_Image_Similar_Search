from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import logging
from supabase import create_client, Client
import chromadb

# Routers
from app.routes.health import router as health_router
from app.routes.metadata_search import router as metadata_search_router
from app.routes.search import router as search_router
from app.routes.task_status import router as task_status_router
from app.routes.upload_embeddings import router as upload_embeddings_router
from app.celery_status import router as celery_status_router

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.getenv("SUPABASE_URL") is None or os.getenv("SUPABASE_KEY") is None:
    logger.error("SUPABASE_URL, and SUPABASE_KEY must be set in environment variables.")
    raise ValueError("SUPABASE_URL, and SUPABASE_KEY must be set")

# Initialize Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Initialize ChromaDB client
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma")
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)

app = FastAPI(title="Textile Design Vector Search API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router)
app.include_router(metadata_search_router)
app.include_router(search_router)
app.include_router(task_status_router)
app.include_router(upload_embeddings_router)
app.include_router(celery_status_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logging.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
