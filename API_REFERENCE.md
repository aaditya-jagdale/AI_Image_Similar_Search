# Fabric Visual Search API Reference

Base URL: `http://localhost:8000`

---

## Endpoints

### 1. Search for Similar Fabrics

**POST** `/api/v1/search`

Search for similar fabric designs using an image.

**cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_encoded_image_or_url>",
    "top_k": 10,
    "min_similarity": 0.4,
    "filters": {
      "gsm_min": 100,
      "gsm_max": 200,
      "width_min": 56,
      "width_max": 65,
      "stock_available": true
    }
  }'
```

**Response:**
```json
{
  "results": [
    {
      "design_no": "JJT/100",
      "similarity_score": 0.950,
      "image_path": "images/JJT-100.jpeg",
      "width": "60",
      "stock": 500,
      "gsm": 165,
      "source_pdf": "JJT FLANNEL.pdf",
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
```

---

### 2. Build Search Index

**POST** `/api/v1/index/build`

Build or rebuild the search index from fabric catalog.

**cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/index/build" \
  -H "Content-Type: application/json" \
  -d '{
    "rebuild": true
  }'
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "total": 533,
    "indexed": 532,
    "failed": 1,
    "skipped": 0
  },
  "message": "Successfully indexed 532 fabrics"
}
```

---

### 3. Get Index Stats

**GET** `/api/v1/index/stats`

Get statistics about the current search index.

**cURL:**
```bash
curl -X GET "http://localhost:8000/api/v1/index/stats"
```

**Response:**
```json
{
  "catalog_count": 532,
  "index_count": 532,
  "status": "ready"
}
```

---

### 4. Health Check

**GET** `/api/v1/health`

Check if the API is running and index is ready.

**cURL:**
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Response:**
```json
{
  "status": "healthy",
  "index_ready": true,
  "index_count": 532
}
```

---

### 5. Serve Fabric Images

**GET** `/images/{filename}`

Retrieve fabric images from the catalog.

**cURL:**
```bash
curl -X GET "http://localhost:8000/images/JJT-100.jpeg" --output JJT-100.jpeg
```

---

## Interactive API Docs

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
