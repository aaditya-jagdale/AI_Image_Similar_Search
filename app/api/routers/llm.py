from fastapi import APIRouter, HTTPException, status, Depends
from app.services.llm_services import LLMService
from app.models.query_model import QueryModel

llm_router = APIRouter()

# Service instance - initialized once at module load
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Dependency to get LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


@llm_router.get("/gemini-details")
def gemini_details():
    """Get Gemini model details."""
    return {
        "MODEL": "gemini-2.5-flash-lite"
    }


@llm_router.post("/chat")
async def gemini_generate(
    query: QueryModel,
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Generate a response using Gemini LLM.
    
    Args:
        query: Query model containing the user's question
        llm_service: Injected LLM service instance
    
    Returns:
        Dictionary containing the LLM response
    """
    if not query.query or not query.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty"
        )

    try:
        response = llm_service.generate(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )