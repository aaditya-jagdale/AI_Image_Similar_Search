from fastapi import APIRouter, HTTPException, status, Depends
from app.services.supabase_service import SupabaseService
from pydantic import BaseModel

supabase_router = APIRouter()

# Service instance - initialized once at module load
_supabase_service: SupabaseService | None = None


def get_supabase_service() -> SupabaseService:
    """Dependency to get Supabase service instance."""
    global _supabase_service
    if _supabase_service is None:
        _supabase_service = SupabaseService()
    return _supabase_service


class IdsModel(BaseModel):
    ids: list[str]


@supabase_router.get("/item/{item_id}")
def get_item(
    item_id: str,
    supabase_service: SupabaseService = Depends(get_supabase_service)
):
    """
    Get a single item by ID from Supabase.
    
    Args:
        item_id: The ID of the item to retrieve
        supabase_service: Injected Supabase service instance
    
    Returns:
        Item data from Supabase
    """
    if not item_id or not item_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="item_id must not be empty"
        )
    
    try:
        response = supabase_service.get_item_by_id(item_id)
        if not response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item with ID {item_id} not found"
            )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving item: {str(e)}"
        )


@supabase_router.post("/items")
def get_items(
    id_list: IdsModel,
    supabase_service: SupabaseService = Depends(get_supabase_service)
):
    """
    Get multiple items by IDs from Supabase.
    
    Args:
        id_list: Model containing list of item IDs
        supabase_service: Injected Supabase service instance
    
    Returns:
        List of items from Supabase
    """
    if not id_list.ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ids list must not be empty"
        )
    
    try:
        response = supabase_service.get_multiple_items_by_ids(id_list.ids)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving items: {str(e)}"
        )


@supabase_router.get("/all_items")
def get_all_items(
    limit: int = 100,
    supabase_service: SupabaseService = Depends(get_supabase_service)
):
    """
    Get all items from Supabase with a limit.
    
    Args:
        limit: Maximum number of items to return (default: 100, max: 1000)
        supabase_service: Injected Supabase service instance
    
    Returns:
        List of items from Supabase
    """
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be between 1 and 1000"
        )
    
    try:
        response = supabase_service.get_all_items(limit)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving items: {str(e)}"
        )


@supabase_router.get("/get_items_by_source")
def get_items_by_source(
    source_pdf: str | None = None,
    limit: int = 12,
    supabase_service: SupabaseService = Depends(get_supabase_service)
):
    """
    Get items filtered by source PDF from Supabase.
    
    Args:
        source_pdf: Optional source PDF name to filter by
        limit: Maximum number of items to return (default: 12, max: 1000)
        supabase_service: Injected Supabase service instance
    
    Returns:
        List of items from Supabase
    """
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be between 1 and 1000"
        )
    
    try:
        response = supabase_service.get_items_by_source(source_pdf, limit)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving items: {str(e)}"
        )