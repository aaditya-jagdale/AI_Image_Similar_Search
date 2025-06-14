from pydantic import BaseModel

class SearchRequest(BaseModel):
    image_url: str
    top_k: int = 10

class MetadataSearchRequest(BaseModel):
    design_no: str = ""
    width: str = ""
    stock: str = ""
    GSM: str = ""
    source_pdf: str = ""
