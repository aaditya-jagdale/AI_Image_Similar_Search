"""
FastAPI Application - Fabric Visual Search API

Phase 1: Quick Win Implementation
- Image-based search using perceptual hashing
- Three-stage filtering pipeline
- SQLite index storage
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from pathlib import Path

from .schemas import (
    SearchRequest, SearchResponse, SearchResult, MatchDetails,
    IndexBuildRequest, IndexBuildResponse, IndexStatsResponse, HealthResponse
)
from .core.image_processor import ImageProcessor
from .core.database import Database
from .core.search_engine import SearchEngine


# Global instances
processor: ImageProcessor = None
db: Database = None
search_engine: SearchEngine = None


# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
CATALOG_JSON = OUTPUT_DIR / "pdf_extracted_data.json"
DB_PATH = DATA_DIR / "fabric_search.db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global processor, db, search_engine
    
    print("Initializing Fabric Visual Search API...")
    
    # Initialize components
    processor = ImageProcessor()
    db = Database(str(DB_PATH))
    search_engine = SearchEngine(db, processor)
    
    # Load index if exists
    index_count = db.get_index_count()
    print(f"Index loaded with {index_count} fabrics")
    
    yield
    
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Fabric Visual Search API",
    description="Search for similar fabric designs using image-based queries",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for images
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    index_count = db.get_index_count() if db else 0
    return HealthResponse(
        status="healthy",
        index_ready=index_count > 0,
        index_count=index_count
    )


@app.post("/api/v1/search", response_model=SearchResponse)
async def search_fabrics(request: SearchRequest):
    """
    Search for similar fabrics using an image.
    
    Accepts:
    - Base64 encoded image data
    - Image URL (http:// or https://)
    
    Returns ranked list of similar fabrics with similarity scores.
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    # Check if index has data
    index_count = db.get_index_count()
    if index_count == 0:
        raise HTTPException(
            status_code=400, 
            detail="Search index is empty. Please build the index first using POST /api/v1/index/build"
        )
    
    # Convert filters to dict if present
    filters = request.filters.model_dump() if request.filters else None
    
    # Perform search
    result = search_engine.search(
        query_source=request.image,
        top_k=request.top_k,
        min_similarity=request.min_similarity,
        filters=filters,
        use_clahe=request.use_clahe
    )
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    # Convert to response model
    search_results = []
    for r in result['results']:
        search_results.append(SearchResult(
            design_no=r['design_no'],
            similarity_score=r['similarity_score'],
            image_path=r['image_path'],
            width=r.get('width'),
            stock=r.get('stock'),
            gsm=r.get('gsm'),
            source_pdf=r.get('source_pdf'),
            match_details=MatchDetails(**r['match_details'])
        ))
    
    return SearchResponse(
        results=search_results,
        total_found=result['total_found'],
        processing_time_ms=result['processing_time_ms']
    )


@app.post("/api/v1/index/build", response_model=IndexBuildResponse)
async def build_index(request: IndexBuildRequest):
    """
    Build or rebuild the search index from the fabric catalog.
    
    This processes all images and extracts features for similarity search.
    May take a few minutes for large catalogs.
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    if not CATALOG_JSON.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Catalog JSON not found at {CATALOG_JSON}"
        )
    
    print(f"Building index from {CATALOG_JSON}...")
    print(f"Images directory: {OUTPUT_DIR}")
    
    try:
        stats = search_engine.index_manager.build_index(
            catalog_json_path=str(CATALOG_JSON),
            images_base_path=str(OUTPUT_DIR),
            rebuild=request.rebuild
        )
        
        return IndexBuildResponse(
            success=True,
            stats=stats,
            message=f"Successfully indexed {stats['indexed']} fabrics"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index build failed: {str(e)}")


@app.get("/api/v1/index/stats", response_model=IndexStatsResponse)
async def get_index_stats():
    """Get statistics about the search index."""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    stats = search_engine.index_manager.get_index_stats()
    
    return IndexStatsResponse(
        catalog_count=stats['catalog_count'],
        index_count=stats['index_count'],
        status="ready" if stats['index_count'] > 0 else "empty"
    )


# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
