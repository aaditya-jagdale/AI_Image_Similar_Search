from pydantic import BaseModel

class AddDocumentModels(BaseModel):
    image_uri: str
    metadata: dict | None = None
    id: str | None = None