from supabase import create_client, Client, AsyncClientOptions
import os
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()


# {'product_id': 'TXT008', 'name': 'asd', 'material': 'Wool', 'pattern': 'Striped', 'type': 'Non-woven', 'gsm': 200, 'price_per_meter': 50.0, 'image_url': 'https://jxlsranqkfbwlwdchtft.supabase.co/storage/v1/object/public/textile-images/0b54dd49-abcc-41b5-90e4-618cd5e519f0/1749715414005-brhqcoxt97t.png', 'created_at': '2025-06-12T08:04:59.370519+00:00', 'updated_at': '2025-06-12T10:49:15.728157+00:00', 'user_id': '0b54dd49-abcc-41b5-90e4-618cd5e519f0', 'id': '719dd3f4-58f8-4313-952c-f42816dc5555', 'price_currency': 'USD', 'price_local': 50.0}

class TextileProduct(BaseModel):
    product_id: str
    name: str
    material: str
    pattern: str
    type: str
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

if __name__ == "__main__":
    #login with example@
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    response = supabase.auth.sign_in_with_password({
        "email": "aadi@gmail.com",
        "password": "123456"
    })
    access_token = response.session.access_token
    print(access_token)
    client = SupabaseClient(access_token)
    res = client.get_all_items()
    for item in res:
        print("Item name: ", item.name)