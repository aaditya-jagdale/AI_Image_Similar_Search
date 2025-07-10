from supabase import create_client, Client, AsyncClientOptions
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
load_dotenv()

class TextileProduct(BaseModel):
    product_id: str
    name: str
    material: str
    pattern: str
    type: Optional[str] = None
    gsm: int
    price_per_meter: float
    image_url: str
    created_at: str
    updated_at: str
    user_id: str
    id: str
    price_currency: str
    price_local: float

class SupabaseClient:
    def __init__(self, access_token: str):
        self.supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"), options=AsyncClientOptions(headers={"Authorization": f"Bearer {access_token}"}))

    def get_all_items(self) -> list[TextileProduct]:
        response = self.supabase.table("textile_products").select("*").execute()
        return [TextileProduct(**item) for item in response.data]
    
    def insert_item(self, item: dict):
        response = self.supabase.table("textile_products").insert({
            "product_id": item["product_id"],
            "name": item["name"],
            "material": item["material"],
            "pattern": item["pattern"],
            "gsm": item["gsm"],
            "price_per_meter": item["price_per_meter"],
            "price_local": item["price_local"],
            "image_url": item["image_url"],
        }).execute()
        return response.data

    def get_company_data(self) -> dict:
        response = self.supabase.table("business_profiles").select("*").execute()
        return response.data

if __name__ == "__main__":
    #login with example@
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    response = supabase.auth.sign_in_with_password({
        "email": "1234@gmail.com",
        "password": "123456"
    })
    access_token = response.session.access_token
    print(access_token)
    # client = SupabaseClient(access_token)
    # res = client.get_all_items()
    # for item in res:
    #     print("Item name: ", item.name)