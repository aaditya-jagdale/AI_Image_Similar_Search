from fastapi import APIRouter
from app.services.supabase_service import SupabaseService
from pydantic import BaseModel

class IdsModel(BaseModel):
    ids: list[str]


supabaseService =  SupabaseService()
supabase_router = APIRouter()


@supabase_router.get("/item/{item_id}")
def get_item(item_id: str):
    response = supabaseService.get_item_by_id(item_id)
    return response

@supabase_router.post("/items")
def get_items(id_list: IdsModel):
    response = supabaseService.get_multiple_items_by_ids(id_list.ids)
    return response

@supabase_router.get("/all_items")
def get_all_items(limit: int = 100):
    response = supabaseService.get_all_items(limit)
    return response