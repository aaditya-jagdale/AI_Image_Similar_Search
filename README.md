# 🧵 Fabric Visual Search System

A high-performance **FastAPI-based** backend designed for intelligent fabric discovery. Upload a reference image, and the system finds visually similar fabric designs from your catalog in milliseconds.

![alt text](<output.png>)

---

## 🚀 Key Features

*   **Visual Search**: Search fabrics using image-based queries.
*   **Multi-Stage Search Engine**: Combines Perceptual Hashing, Color Histograms, and ORB Feature Matching for high accuracy.
*   **Metadata Filtering**: Narrow down results by GSM, width, and stock availability.
*   **Blazing Fast**: Response times under 500ms for catalogs with hundreds of designs.
*   **Phased Evolution**: Built to scale from a quick hashing prototype to a full ML-powered vector search system.

---

## 🏗️ System Architecture

The project is built with a modular architecture focused on speed and modularity:

1.  **FastAPI Backend**: Handles requests, validation (Pydantic), and serves the API.
2.  **Image Processor**: 
    - **Normalization**: Resizing and color correction.
    - **Feature Generation**: Computes multiple perceptual hashes (pHash, dHash, aHash, wHash).
    - **Color Analysis**: Extracts color distributions in LAB/HSV space.
3.  **Search & Ranking Engine**: 
    - **Stage 1**: Fast bitwise filtering using Hamming distance on perceptual hashes.
    - **Stage 2**: Color similarity refinement using Histogram comparisons.
    - **Stage 3**: Geometric validation via ORB feature matching.
4.  **Storage Layer**: Manages fabric metadata and pre-calculated image features for instant retrieval.

---

## 🎨 End Product: Visual Experience

The end product provides a seamless way for sales teams and customers to match fabrics. Integrated into a frontend, it offers:

*   **Instant Visual Feedback**: Upload a photo from a phone or catalog and immediately see visual matches.
*   **Ranked Results**: View matches sorted by similarity percentage (e.g., *"95% Match Found"*).
*   **Rich Context**: Alongside the image, you get instant access to critical specs like **Stock Levels**, **GSM**, and **Design Numbers**.

---

## 🛠️ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file based on `.env.example`.

### 3. Run the API
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs` for the interactive API documentation.

### 4. Index Your Catalog
```bash
curl -X POST "http://localhost:8000/api/v1/index/build" -H "Content-Type: application/json" -d '{"rebuild": true}'
```

---

## 📁 Project Structure

```text
├── app/
│   ├── core/           # Core logic (search engine, image processor)
│   ├── main.py         # API entry point
│   └── schemas.py      # Data models
├── data/               # Catalog images & database
├── API_REFERENCE.md    # Detailed API documentation
└── PRD.md              # Vision and roadmaps
```

---
*Developed for efficient textile and fashion inventory management.*