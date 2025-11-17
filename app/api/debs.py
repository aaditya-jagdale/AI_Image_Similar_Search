from fastapi import APIRouter
from app.api.routers import health, image_search, llm, supabase_router
router = APIRouter()

# include routers instead of using decorators (decorators require a following function/class)
router.include_router(health.router)
router.include_router(llm.llm_router, prefix="/llm", tags=["llm"])
router.include_router(image_search.image_search_router, prefix="/image", tags=["Image Search"])
router.include_router(supabase_router.supabase_router, prefix="/supabase", tags=["Supabase"])