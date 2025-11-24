from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.debs import router
from app.startup.services import initialize_all_services, cleanup_services
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting up application...")
    initialize_all_services()
    yield
    # Shutdown
    logger.info("Shutting down application...")
    cleanup_services()


# API App with lifespan events
app = FastAPI(
    title="AI Image Similar Search",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Prefix '/api' endpoint
app.include_router(router=router, prefix="/api")