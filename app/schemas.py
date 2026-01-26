"""
Pydantic Schemas for API Request/Response Validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class SearchFilters(BaseModel):
    """Filters for search queries."""
    gsm_min: Optional[int] = Field(None, description="Minimum GSM value")
    gsm_max: Optional[int] = Field(None, description="Maximum GSM value")
    width_min: Optional[float] = Field(None, description="Minimum width in inches")
    width_max: Optional[float] = Field(None, description="Maximum width in inches")
    stock_available: Optional[bool] = Field(False, description="Only show in-stock items")


class SearchRequest(BaseModel):
    """Request body for image search."""
    image: str = Field(..., description="Base64 encoded image or URL")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    min_similarity: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")
    filters: Optional[SearchFilters] = Field(None, description="Optional metadata filters")
    use_clahe: bool = Field(True, description="Use CLAHE preprocessing for better external image matching")


class MatchDetails(BaseModel):
    """Detailed breakdown of similarity scores."""
    hash_score: float
    color_score: float
    feature_score: float
    feature_matches: int


class SearchResult(BaseModel):
    """Single search result."""
    design_no: str
    similarity_score: float
    image_path: str
    width: Optional[str] = None
    stock: Optional[int] = None
    gsm: Optional[int] = None
    source_pdf: Optional[str] = None
    match_details: MatchDetails


class SearchResponse(BaseModel):
    """Response from search endpoint."""
    results: List[SearchResult]
    total_found: int
    processing_time_ms: int
    error: Optional[str] = None


class IndexBuildRequest(BaseModel):
    """Request to build/rebuild search index."""
    rebuild: bool = Field(False, description="If true, clear existing index first")


class IndexBuildResponse(BaseModel):
    """Response from index build."""
    success: bool
    stats: Dict[str, int]
    message: str


class IndexStatsResponse(BaseModel):
    """Response with index statistics."""
    catalog_count: int
    index_count: int
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    index_ready: bool
    index_count: int
