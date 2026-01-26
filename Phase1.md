Phase 1: Fabric Visual Search - Completion Walkthrough
Summary
Successfully implemented a FastAPI-based visual search system for fabric designs using perceptual hashing and a three-stage filtering pipeline.

What Was Built
Architecture
app/
├── __init__.py
├── main.py              # FastAPI app + endpoints
├── schemas.py           # Pydantic models
└── core/
    ├── __init__.py
    ├── image_processor.py   # Preprocessing + feature extraction
    ├── search_engine.py     # Three-stage search pipeline
    └── database.py          # SQLite operations
Core Components
Component	File	Purpose
Image Processor	
image_processor.py
Preprocessing, multi-hash generation, color histograms, ORB features
Database	
database.py
SQLite storage for catalog and search index
Search Engine	
search_engine.py
Three-stage filtering pipeline with weighted scoring
API	
main.py
FastAPI endpoints for search, indexing, health check
Algorithm Implementation
Three-Stage Filtering Pipeline
Query Image
Stage 1: Hash Filter
Stage 2: Color Filter
Stage 3: Feature Match
Final Ranking
Top-K Results
Stage 1: Perceptual Hash Filtering

Generates pHash, dHash, aHash, wHash (16x16 = 256 bits)
Filters candidates with Hamming distance ≤ 80 bits (~31% difference)
Very fast: processes 532 images in milliseconds
Stage 2: Color Histogram Similarity

LAB color space histograms (32 bins per channel)
Correlation-based comparison
Threshold: 0.3 minimum similarity
Stage 3: ORB Feature Matching

500 ORB keypoints per image
KNN matching with Lowe's ratio test (0.75)
Geometric verification
Final Scoring Weights:

Hash similarity: 40%
Color similarity: 30%
Feature matches: 20%
Metadata bonus: 10%
API Endpoints
Endpoint	Method	Description
/api/v1/search	POST	Search for similar fabrics
/api/v1/index/build	POST	Build/rebuild search index
/api/v1/index/stats	GET	Get index statistics
/api/v1/health	GET	Health check
/images/{filename}	GET	Serve fabric images
Example Search Request
POST /api/v1/search
{
  "image": "<base64_encoded_image>",
  "top_k": 10,
  "min_similarity": 0.4,
  "filters": {
    "gsm_min": 100,
    "gsm_max": 200
  }
}
Example Response
{
  "results": [
    {
      "design_no": "JJT/100",
      "similarity_score": 0.950,
      "image_path": "images/JJT-100.jpeg",
      "width": "60",
      "stock": 500,
      "gsm": 165,
      "match_details": {
        "hash_score": 1.000,
        "color_score": 1.000,
        "feature_score": 1.000,
        "feature_matches": 421
      }
    }
  ],
  "total_found": 14,
  "processing_time_ms": 279
}
Verification Results
Test Summary
Test	Result	Details
Cotton Stripe Search	✅ PASS	Found 1 match (exact)
Chambray Search	✅ PASS	Found 2 matches (JCY/1, JCY/3)
Heavy GSM Print Search	✅ PASS	Found 2 matches
GSM Filter	✅ PASS	Correctly filters by metadata
Performance	✅ PASS	Avg: 26ms, Max: 37ms (target: <3000ms)
Performance Metrics
Metric	Target	Actual
Response Time (avg)	< 3000ms	26ms
Response Time (max)	< 3000ms	37ms
Index Build Time	-	~60 seconds for 532 images
Database Size	-	~8MB
How to Run
# Start the server
cd /Users/aaditya/backend/AI_Image_Similar_Search
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Build index (first time)
curl -X POST "http://localhost:8000/api/v1/index/build" \
  -H "Content-Type: application/json" \
  -d '{"rebuild": true}'
# Check health
curl http://localhost:8000/api/v1/health
Index Statistics
Total fabrics indexed: 532
Categories: Stripes (JST), Chambrays (JCY), Flannels (JJT), Prints (JCP), etc.
Source: PDF extracted data from 
/data/output/pdf_extracted_data.json
Known Limitations (Phase 1)
Single-threaded processing
In-memory index caching (reloads from SQLite on restart)
Basic preprocessing (no perspective correction)
No caching of search results
Hash threshold may need tuning for specific use cases
Next Steps (Phase 2)
PostgreSQL + pgvector for production
Redis caching layer
Advanced preprocessing (perspective correction, CLAHE)
Texture feature extraction (LBP, Gabor filters)
Batch processing API
Analytics and monitoring