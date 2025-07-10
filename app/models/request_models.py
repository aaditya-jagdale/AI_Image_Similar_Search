from pydantic import BaseModel
from typing import List

class SearchRequest(BaseModel):
    image_url: str
    top_k: int = 10

class MetadataSearchRequest(BaseModel):
    design_no: str = ""
    width: str = ""
    stock: str = ""
    GSM: str = ""
    source_pdf: str = ""

class ChatMessage(BaseModel):
    role: str
    message: str

class TextChatRequest(BaseModel):
    messages: List[ChatMessage]


class ProductModels(BaseModel):
    material: str | None = None
    gsm: int | None = None
    price_per_meter: float | None = None
    product_id: str | None = None
    updated_at: str | None = None
    price_currency: str | None = None
    image_url: str | None = None
    price_local: float | None = None
    id: str | None = None
    user_id: str | None = None
    created_at: str | None = None
    pattern: str | None = None
    name: str | None = None
    type: str | None = None