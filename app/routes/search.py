from fastapi import APIRouter, Depends, HTTPException, Header
from app.models.request_models import SearchRequest, TextChatRequest
from app.utils.auth import get_current_user
from app.services.search_service import perform_search
import logging
from app.routes.text_chat import perform_text_chat
from app.supabase_client import SupabaseClient
import asyncio
from app.supabase_admin import fetch_textile_items
router = APIRouter()
logger = logging.getLogger(__name__)

def get_company_data(access_token: str):
    # get company data from supabase
    supabase = SupabaseClient(access_token)
    response = supabase.get_company_data()
    return response[0]


@router.post("/search")
async def search_route(request: SearchRequest, user_id: str = Depends(get_current_user)):
    try:
        logger.info(f"🔍 Starting search for image_url: {request.image_url}, top_k: {request.top_k}")
        logger.info(f" User ID: {user_id}")
        results = perform_search(user_id, request.image_url, request.top_k)
        logger.info("✅ Search completed")
        if not results:
            logger.warning("Search returned no results or an error occurred upstream.")
            return {"status": "ok", "results": []}
    except HTTPException as e:
        logger.error(f"HTTPException during search: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed due to an internal error: {e}")
    return {"status": "ok", "results": results}

@router.post("/text_chat")
async def text_chat_route(
    request: TextChatRequest,
    authorization: str = Header(..., alias="Authorization"),
    user_id: str = Depends(get_current_user),
):
    try:
        # Extract the raw JWT (remove the "Bearer " prefix)
        try:
            token = authorization.split(" ")[1]
        except IndexError:
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        company_data = get_company_data(token)
        for m in reversed(request.messages):
            if m.role.lower() == "user":
                latest_user_message = m.message
                break

        # results = await perform_text_chat(request, company_data=company_data, user_id=user_id)
        results = fetch_textile_items(user_id, latest_user_message)
        logger.info("✅ Text chat completed")
        return {"status": "ok", "message": f"Found {len(results['products'])} items", "items" : results['products']}
    except HTTPException as e:
        logger.error(f"HTTPException during text chat: {e.detail}")
        raise
