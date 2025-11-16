from fastapi import APIRouter
from app.api.routers import health, gemini_api
router = APIRouter()

# include routers instead of using decorators (decorators require a following function/class)
router.include_router(health.router)
router.include_router(gemini_api.llm_router, prefix="/llm", tags=["llm"])