from fastapi import APIRouter
from app.services.llm_services import LLMService
from app.models.query_model import QueryModel
from fastapi import HTTPException, status

llm_router = APIRouter()
gemini = LLMService()

@llm_router.get("/gemini-details")
def gemini_details():
    return {
        "MODEL": "model"
    }

@llm_router.post("/chat")
async def gemini_generate(query: QueryModel):
    if not query.query or not query.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty"
        )

    response = gemini.generate(query)
    return {"response": response}