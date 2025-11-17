import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

class SupabaseService:
    def __init__(self):
        self.client = supabase

    def get_item_by_id(self, item_id: str):
        response = self.client.table("products").select("*").eq("id", item_id).execute()
        return response.data
    
    def get_multiple_items_by_ids(self, ids: list[str]):
        response = self.client.table("products").select("*").in_("id", ids).execute()
        return response.data
    
    def get_all_items(self, limit: int = 100):
        response = self.client.table("products").select("*").limit(limit).execute()
        return response.data