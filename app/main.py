from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.debs import router
from pathlib import Path
import logging
from contextlib import asynccontextmanager

# download helper
from app.startup.download_models import download_models_from_env

@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for FastAPI startup/shutdown.

    This will be called by FastAPI on application startup. We keep the
    same behavior as before (download models if environment variables are
    set) but implement the proper signature and yield control back to the
    server so the app starts normally.
    """
    logger = logging.getLogger("app.startup")
    try:
        project_root = Path(__file__).resolve().parents[1]
        logger.info("Startup: ensure ONNX model files are present in %s", project_root)
        download_models_from_env(project_root)
    except Exception:
        logger.exception("Failed during startup model download step")
    # Hand control back to FastAPI to continue startup
    try:
        yield
    finally:
        # optional: any shutdown cleanup could go here
        logger.info("Shutdown: completed lifespan cleanup")

#API App
app = FastAPI(title="AI Image Similar Search", version="1.0.0", lifespan=lifespan)

#Middleware for security
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["POST", "GET"],
    allow_headers = ["*"],
)

#Prefix '/api' endpoint
app.include_router(router=router, prefix="/api")