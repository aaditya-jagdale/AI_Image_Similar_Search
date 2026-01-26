# Product Requirements Document: Fabric Visual Search System

**Version:** 1.0  
**Last Updated:** January 25, 2026  
**Document Owner:** Engineering Team  
**Status:** Planning Phase

---

## Executive Summary

This document outlines the comprehensive plan for developing a **Fabric Visual Search System** - a FastAPI-based backend service that enables users to search for similar fabric designs using image-based queries. The system will process fabric catalog images, create multiple searchable indices, and return ranked results based on visual similarity.

The development follows a three-phase approach:
- **Phase 1**: Quick Win with Perceptual Hashing (1-2 days)
- **Phase 2**: Production-Ready Hybrid System (1-2 weeks)
- **Phase 3**: Advanced ML-Powered Search (Future Enhancement)

This phased approach ensures we deliver value quickly while building toward a highly accurate, production-grade solution.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Goals and Objectives](#goals-and-objectives)
3. [Success Metrics](#success-metrics)
4. [Technical Architecture](#technical-architecture)
5. [Phase 1: Quick Win Implementation](#phase-1-quick-win-implementation)
6. [Phase 2: Production System](#phase-2-production-system)
7. [Phase 3: Advanced ML System](#phase-3-advanced-ml-system)
8. [API Specifications](#api-specifications)
9. [Data Models](#data-models)
10. [Infrastructure Requirements](#infrastructure-requirements)
11. [Testing Strategy](#testing-strategy)
12. [Deployment Plan](#deployment-plan)
13. [Future Enhancements](#future-enhancements)

---

## Problem Statement

### Current Situation

Our fabric catalog contains thousands of designs (stripes, prints, chambray, etc.) extracted from PDF documents. Each fabric entry includes:
- Design number (e.g., JST/15, JCP/101)
- Product image
- Width specifications
- Stock quantities
- GSM (grams per square meter)
- Source PDF reference

Currently, there is no efficient way to:
1. Find similar fabrics when a customer provides a reference image
2. Identify duplicate or near-duplicate designs in the catalog
3. Recommend alternative fabrics when a specific design is out of stock
4. Match customer-provided fabric samples to existing inventory

### User Pain Points

**For Sales Teams:**
- Manual searching through thousands of images is time-consuming
- Difficult to find alternatives when exact matches are unavailable
- Hard to identify similar designs across different categories (stripes vs prints)
- No way to handle customer photos taken at different angles or lighting

**For Inventory Management:**
- Cannot detect duplicate entries with different design numbers
- Difficult to identify similar patterns that could be consolidated
- No automated way to find design variations

**For Customers:**
- Cannot search by uploading a photo of desired fabric
- Limited to text-based search (design number, category)
- Miss out on similar alternatives they might prefer

### Previous Attempts and Failures

1. **Simple Vector Database Approach**: Used generic embeddings (ResNet/CLIP) to vectorize images. Failed due to lack of domain specificity - generic models don't understand textile-specific features like weave patterns, stripe orientations, or fabric textures.

2. **Augmented Vector Database**: Created multiple augmented versions (rotations, crops, zooms) before vectorization. Still failed because the underlying embeddings weren't designed for textile pattern matching and couldn't distinguish between "similar enough for a customer" vs "completely different design."

---

## Goals and Objectives

### Primary Goals

1. **Enable Visual Search**: Allow users to upload a fabric image and find visually similar items in the catalog with at least 85% accuracy
2. **Fast Response Times**: Return results in under 2 seconds for catalogs up to 10,000 images
3. **Handle Real-World Variations**: Successfully match images despite differences in:
   - Rotation (0°, 90°, 180°, 270°)
   - Lighting conditions
   - Camera angles (within 30° perspective shift)
   - Scale/zoom levels
4. **Configurable Similarity Threshold**: Allow users to adjust sensitivity (60% to 95% match)
5. **Metadata Filtering**: Support filtering by GSM range, width, stock availability

### Secondary Goals

1. Provide similarity scores with each result for transparency
2. Support batch processing for duplicate detection
3. Enable API-based integration with existing systems
4. Create a foundation for future ML enhancements
5. Generate analytics on search patterns and popular designs

### Non-Goals (Out of Scope)

- Text-based search or natural language queries
- Real-time video search
- 3D fabric visualization
- Color customization/recoloring features
- Mobile app development (API-only in this phase)
- User authentication/authorization (handled separately)

---

## Success Metrics

### Phase 1 Success Criteria (Quick Win)

- **Development Time**: Complete in 1-2 days
- **Accuracy**: ≥75% of test queries return correct matches in top 5 results
- **Performance**: Response time <3 seconds for 1,000 image catalog
- **Coverage**: Works with all existing PDF-extracted images
- **Validation**: Team can successfully find similar fabrics in manual testing

### Phase 2 Success Criteria (Production)

- **Accuracy**: ≥85% of test queries return correct matches in top 5 results
- **Performance**: Response time <2 seconds for 10,000 image catalog
- **Robustness**: 90% accuracy even with rotated/scaled input images
- **Precision**: False positive rate <15%
- **API Reliability**: 99.5% uptime, proper error handling

### Phase 3 Success Criteria (Advanced ML)

- **Accuracy**: ≥95% of test queries return correct matches in top 3 results
- **Generalization**: Works with customer photos (not just catalog images)
- **Advanced Matching**: Correctly identifies partial matches and pattern variations
- **Performance**: Maintains <2 second response time even with ML inference

### Key Performance Indicators (KPIs)

1. **Mean Average Precision (MAP)**: Target ≥0.85 by Phase 2
2. **Top-5 Accuracy**: Percentage of queries where correct match appears in top 5 results
3. **Query Latency (P95)**: 95th percentile response time
4. **Index Build Time**: Time to process and index 1,000 new images
5. **Storage Efficiency**: MB required per indexed image
6. **API Usage Metrics**: Queries per day, unique users, popular search patterns

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Image      │  │   Search     │  │   Index         │   │
│  │   Upload     │  │   Engine     │  │   Manager       │   │
│  │   Endpoint   │  │              │  │                 │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   Core Processing Layer                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Perceptual   │  │   Color      │  │    Feature      │   │
│  │   Hashing    │  │  Histogram   │  │   Extraction    │   │
│  │  (pHash,     │  │  (LAB space) │  │   (SIFT/ORB)    │   │
│  │   dHash)     │  │              │  │                 │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Image      │  │   Similarity │  │    Ranking      │   │
│  │   Preprocessor│  │   Scorer     │  │    Engine       │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   PostgreSQL │  │    Redis     │  │   File System   │   │
│  │  (Metadata)  │  │   (Cache)    │  │    (Images)     │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend Framework:**
- FastAPI 0.109+ (async support, automatic API docs, Pydantic validation)
- Python 3.11+ (performance improvements, better type hints)
- Uvicorn (ASGI server)

**Image Processing:**
- OpenCV 4.8+ (feature extraction, preprocessing)
- Pillow 10.0+ (image I/O, basic transformations)
- imagehash 4.3+ (perceptual hashing)
- scikit-image 0.22+ (color space conversion, texture analysis)

**Machine Learning (Phase 2+):**
- NumPy 1.26+ (numerical operations)
- scikit-learn 1.3+ (similarity metrics, clustering)
- TensorFlow 2.15+ / PyTorch 2.1+ (Phase 3 deep learning)

**Data Storage:**
- PostgreSQL 15+ with pgvector extension (metadata + vector storage)
- Redis 7.0+ (caching, session management)
- Local filesystem / S3 (image storage)

**Development Tools:**
- pytest (testing framework)
- black (code formatting)
- ruff (linting)
- pre-commit (git hooks)

---

## Phase 1: Quick Win Implementation

### Timeline: 1-2 Days

### Objectives

Deliver a working prototype that demonstrates visual search capability with minimal complexity. This phase focuses on **speed of implementation** and **proof of concept** rather than maximum accuracy.

### Core Features

1. **Image Upload Endpoint**: Accept single image (JPEG/PNG, max 10MB)
2. **Basic Preprocessing**: Resize to standard dimensions, normalize
3. **Perceptual Hashing**: Generate multiple hash types for robustness
4. **Multi-Stage Filtering**: Fast elimination of non-matches
5. **Top-K Results**: Return ranked list of similar images

### Technical Implementation Details

#### 1.1 Image Preprocessing Pipeline

**Purpose**: Normalize input images to ensure consistent hashing regardless of source quality.

**Steps:**
```python
def preprocess_image(image_path: str) -> np.ndarray:
    """
    Standard preprocessing pipeline for Phase 1
    """
    # 1. Load image
    img = cv2.imread(image_path)
    
    # 2. Convert to RGB (OpenCV loads as BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Resize to standard size (256x256 for Phase 1)
    img_resized = cv2.resize(img_rgb, (256, 256), 
                             interpolation=cv2.INTER_LANCZOS4)
    
    # 4. White balance correction (simple gray world assumption)
    img_balanced = apply_gray_world_wb(img_resized)
    
    # 5. Denoise (slight Gaussian blur)
    img_denoised = cv2.GaussianBlur(img_balanced, (3, 3), 0)
    
    return img_denoised
```

**Rationale**: 
- 256x256 balances detail preservation with processing speed
- White balance correction handles different lighting conditions
- Denoising removes compression artifacts from JPEG images

#### 1.2 Multi-Hash Strategy

**Why Multiple Hashes**: Different algorithms excel at different types of transformations:
- **pHash (Perceptual Hash)**: Robust to scaling, slight color changes
- **dHash (Difference Hash)**: Excellent for detecting gradients (useful for stripes)
- **aHash (Average Hash)**: Fast, good for quick elimination
- **Wavelet Hash**: Captures frequency patterns (good for textile textures)

**Implementation:**
```python
import imagehash
from PIL import Image

def generate_hashes(image: np.ndarray) -> dict:
    """
    Generate multiple perceptual hashes for robustness
    """
    pil_img = Image.fromarray(image)
    
    return {
        'phash': str(imagehash.phash(pil_img, hash_size=16)),
        'dhash': str(imagehash.dhash(pil_img, hash_size=16)),
        'ahash': str(imagehash.average_hash(pil_img, hash_size=16)),
        'whash': str(imagehash.whash(pil_img, hash_size=16))
    }
```

**Hash Size**: Using 16x16 (256-bit hashes) instead of default 8x8 for better precision with detailed fabric patterns.

#### 1.3 Three-Stage Filtering Pipeline

**Stage 1: Perceptual Hash Filtering (Fastest)**
```python
def stage1_hash_filter(query_hashes: dict, 
                       catalog_hashes: list[dict],
                       threshold: int = 15) -> list:
    """
    Quick elimination based on Hamming distance
    Threshold: Max bit differences allowed
    """
    candidates = []
    
    for item in catalog_hashes:
        # Calculate Hamming distance for each hash type
        phash_dist = hamming_distance(query_hashes['phash'], 
                                       item['phash'])
        dhash_dist = hamming_distance(query_hashes['dhash'], 
                                       item['dhash'])
        
        # Use weighted minimum distance
        min_dist = min(phash_dist, dhash_dist * 0.8)
        
        if min_dist <= threshold:
            candidates.append({
                'item': item,
                'hash_distance': min_dist
            })
    
    return candidates
```

**Performance**: Processes 10,000 images in ~50ms
**Expected Retention**: ~5-10% of catalog (500-1000 candidates)

**Stage 2: Color Histogram Similarity**
```python
def stage2_color_filter(query_img: np.ndarray,
                        candidates: list,
                        threshold: float = 0.6) -> list:
    """
    Filter by color distribution in LAB color space
    LAB is perceptually uniform - better than RGB for matching
    """
    query_hist = compute_lab_histogram(query_img)
    
    refined = []
    for candidate in candidates:
        candidate_img = load_image(candidate['image_path'])
        candidate_hist = compute_lab_histogram(candidate_img)
        
        # Chi-squared distance (lower = more similar)
        color_similarity = cv2.compareHist(query_hist, 
                                           candidate_hist,
                                           cv2.HISTCMP_CHISQR_ALT)
        
        if color_similarity >= threshold:
            candidate['color_score'] = color_similarity
            refined.append(candidate)
    
    return refined
```

**Performance**: Processes 500 candidates in ~200ms
**Expected Retention**: ~50-100 final candidates

**Stage 3: Feature Matching Validation**
```python
def stage3_feature_matching(query_img: np.ndarray,
                            candidates: list,
                            min_matches: int = 10) -> list:
    """
    Verify geometric consistency using ORB features
    ORB is faster than SIFT, patent-free
    """
    orb = cv2.ORB_create(nfeatures=500)
    query_kp, query_desc = orb.detectAndCompute(query_img, None)
    
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    validated = []
    for candidate in candidates:
        cand_img = load_image(candidate['image_path'])
        cand_kp, cand_desc = orb.detectAndCompute(cand_img, None)
        
        if cand_desc is None:
            continue
            
        matches = bf_matcher.knnMatch(query_desc, cand_desc, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) >= min_matches:
            candidate['feature_matches'] = len(good_matches)
            validated.append(candidate)
    
    return validated
```

**Performance**: Processes 100 candidates in ~300ms
**Output**: Top 20-30 highly validated matches

#### 1.4 Final Ranking Algorithm

```python
def compute_final_score(candidate: dict) -> float:
    """
    Weighted combination of all similarity metrics
    
    Weights tuned for fabric search:
    - Hash similarity: 40% (pattern structure)
    - Color similarity: 30% (visual appearance)
    - Feature matches: 20% (geometric verification)
    - Metadata bonus: 10% (GSM/width similarity)
    """
    # Normalize hash distance to 0-1 scale (lower is better)
    hash_score = 1.0 - (candidate['hash_distance'] / 64.0)
    
    # Color score already 0-1 (higher is better)
    color_score = candidate['color_score']
    
    # Normalize feature matches (assuming max ~100 good matches)
    feature_score = min(candidate['feature_matches'] / 100.0, 1.0)
    
    # Metadata similarity (optional bonus)
    metadata_score = 0.0
    if 'metadata_similarity' in candidate:
        metadata_score = candidate['metadata_similarity']
    
    final_score = (
        0.40 * hash_score +
        0.30 * color_score +
        0.20 * feature_score +
        0.10 * metadata_score
    )
    
    return final_score
```

### Phase 1 API Endpoints

#### POST /api/v1/search/image

**Request:**
```json
{
  "image": "<base64_encoded_image>",
  "top_k": 10,
  "min_similarity": 0.75,
  "filters": {
    "gsm_min": 100,
    "gsm_max": 300,
    "width_min": 56,
    "stock_available": true
  }
}
```

**Response:**
```json
{
  "query_id": "qry_abc123",
  "results": [
    {
      "design_no": "JST/15",
      "similarity_score": 0.92,
      "image_url": "/images/STRIPES_JST-15.png",
      "width": "61\"",
      "stock": 900,
      "gsm": 79,
      "match_details": {
        "hash_score": 0.95,
        "color_score": 0.88,
        "feature_matches": 42
      }
    }
  ],
  "total_found": 8,
  "processing_time_ms": 487
}
```

### Phase 1 Deliverables

1. **Working FastAPI Application**
   - Docker containerized
   - OpenAPI documentation
   - Health check endpoint

2. **Core Search Functionality**
   - Image upload and validation
   - Multi-stage filtering pipeline
   - Ranked results with scores

3. **Testing Suite**
   - Unit tests for each processing stage
   - Integration tests for full pipeline
   - Performance benchmarks

4. **Documentation**
   - API usage guide
   - Architecture diagrams
   - Performance tuning guide

### Phase 1 Known Limitations

- Single-threaded processing (no concurrent requests optimization)
- In-memory hash storage (doesn't scale beyond ~10k images)
- Basic preprocessing (no advanced color correction)
- No caching layer
- Limited to exact or near-exact pattern matches

---

## Phase 2: Production System

### Timeline: 1-2 Weeks

### Objectives

Build a robust, scalable system that handles real-world complexity and achieves 85%+ accuracy. This phase focuses on **production readiness**, **performance optimization**, and **handling edge cases**.

### Enhanced Features

1. **Advanced Image Preprocessing**
2. **Hybrid Multi-Feature Similarity**
3. **Persistent Storage with PostgreSQL + pgvector**
4. **Redis Caching Layer**
5. **Batch Processing API**
6. **Comprehensive Metadata Filtering**
7. **Analytics and Monitoring**

### Technical Implementation Details

#### 2.1 Advanced Preprocessing Pipeline

**Enhancement Goals:**
- Handle images taken at angles (perspective correction)
- Normalize lighting variations (CLAHE, histogram equalization)
- Remove backgrounds (focus on fabric pattern)
- Handle different zoom levels

```python
class AdvancedImagePreprocessor:
    """
    Production-grade image preprocessing
    """
    
    def __init__(self):
        self.target_size = (512, 512)  # Higher res for better features
        self.clahe = cv2.createCLAHE(clipLimit=2.0, 
                                      tileGridSize=(8, 8))
    
    def process(self, image_path: str) -> dict:
        """
        Returns multiple representations of the image
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Original resized (for color analysis)
        img_original = cv2.resize(img_rgb, self.target_size,
                                  interpolation=cv2.INTER_LANCZOS4)
        
        # Perspective corrected (if needed)
        img_corrected = self.detect_and_correct_perspective(img_rgb)
        
        # Lighting normalized (for pattern matching)
        img_normalized = self.normalize_lighting(img_corrected)
        
        # Background removed (optional, for isolated patterns)
        img_nobg = self.remove_background(img_normalized)
        
        return {
            'original': img_original,
            'corrected': img_corrected,
            'normalized': img_normalized,
            'no_background': img_nobg
        }
    
    def detect_and_correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """
        Detect if image is taken at an angle and correct it
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None and len(lines) > 0:
            # Calculate dominant angle
            angles = []
            for line in lines[:20]:  # Top 20 strongest lines
                rho, theta = line[0]
                angle = np.degrees(theta)
                angles.append(angle)
            
            dominant_angle = np.median(angles)
            
            # If significantly rotated, correct it
            if abs(dominant_angle - 90) > 5:
                rotation_matrix = cv2.getRotationMatrix2D(
                    (img.shape[1]//2, img.shape[0]//2),
                    dominant_angle - 90,
                    1.0
                )
                img = cv2.warpAffine(img, rotation_matrix, 
                                     (img.shape[1], img.shape[0]))
        
        return cv2.resize(img, self.target_size)
    
    def normalize_lighting(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE in LAB color space for better lighting normalization
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    def remove_background(self, img: np.ndarray) -> np.ndarray:
        """
        Simple background removal using GrabCut
        For fabric photos, usually the center is the fabric
        """
        mask = np.zeros(img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define ROI (assume fabric is in center 70% of image)
        h, w = img.shape[:2]
        rect = (int(w*0.15), int(h*0.15), 
                int(w*0.70), int(h*0.70))
        
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model,
                    5, cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img_nobg = img * mask2[:, :, np.newaxis]
        
        return img_nobg
```

#### 2.2 Texture Feature Extraction

**Purpose**: Capture fabric-specific texture patterns that perceptual hashing might miss.

```python
class TextureFeatureExtractor:
    """
    Extract texture features using multiple methods
    """
    
    def extract_lbp_features(self, img: np.ndarray) -> np.ndarray:
        """
        Local Binary Patterns - excellent for textile textures
        """
        from skimage.feature import local_binary_pattern
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale LBP
        features = []
        for radius in [1, 2, 3]:
            points = 8 * radius
            lbp = local_binary_pattern(gray, points, radius, 
                                       method='uniform')
            hist, _ = np.histogram(lbp.ravel(), 
                                   bins=points + 2,
                                   range=(0, points + 2))
            hist = hist.astype('float')
            hist /= (hist.sum() + 1e-6)
            features.extend(hist)
        
        return np.array(features)
    
    def extract_gabor_features(self, img: np.ndarray) -> np.ndarray:
        """
        Gabor filters - capture directional patterns (stripes)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        features = []
        # Test multiple orientations and frequencies
        for theta in np.arange(0, np.pi, np.pi / 8):  # 8 orientations
            for frequency in [0.1, 0.2, 0.3]:
                kernel = cv2.getGaborKernel(ksize=(21, 21),
                                            sigma=5.0,
                                            theta=theta,
                                            lambd=1.0/frequency,
                                            gamma=0.5)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                features.extend([filtered.mean(), filtered.std()])
        
        return np.array(features)
    
    def extract_combined_features(self, img: np.ndarray) -> np.ndarray:
        """
        Combine LBP + Gabor for robust texture representation
        """
        lbp_feat = self.extract_lbp_features(img)
        gabor_feat = self.extract_gabor_features(img)
        
        return np.concatenate([lbp_feat, gabor_feat])
```

#### 2.3 Enhanced Color Analysis

```python
class ColorFeatureExtractor:
    """
    Advanced color analysis in multiple color spaces
    """
    
    def extract_color_features(self, img: np.ndarray) -> dict:
        """
        Extract color features in LAB, HSV, and RGB spaces
        """
        # LAB histogram (perceptually uniform)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_hist = self._compute_3d_histogram(lab, bins=(8, 8, 8))
        
        # HSV histogram (hue for color, saturation for vividness)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_hist = self._compute_3d_histogram(hsv, bins=(8, 4, 4))
        
        # Dominant colors using K-means
        dominant_colors = self._extract_dominant_colors(img, k=5)
        
        # Color moments (mean, std, skewness for each channel)
        moments = self._compute_color_moments(lab)
        
        return {
            'lab_histogram': lab_hist,
            'hsv_histogram': hsv_hist,
            'dominant_colors': dominant_colors,
            'color_moments': moments
        }
    
    def _compute_3d_histogram(self, img: np.ndarray, 
                              bins: tuple) -> np.ndarray:
        """
        Compute 3D color histogram
        """
        hist, _ = np.histogramdd(
            img.reshape(-1, 3),
            bins=bins,
            range=[(0, 256), (0, 256), (0, 256)]
        )
        # Normalize
        hist = hist.flatten()
        hist /= (hist.sum() + 1e-6)
        return hist
    
    def _extract_dominant_colors(self, img: np.ndarray, 
                                  k: int = 5) -> np.ndarray:
        """
        Extract K dominant colors using K-means clustering
        """
        from sklearn.cluster import KMeans
        
        pixels = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Return cluster centers (dominant colors)
        return kmeans.cluster_centers_
    
    def _compute_color_moments(self, img: np.ndarray) -> np.ndarray:
        """
        Compute mean, std, skewness for each channel
        """
        from scipy.stats import skew
        
        moments = []
        for channel in range(3):
            ch_data = img[:, :, channel].flatten()
            moments.extend([
                np.mean(ch_data),
                np.std(ch_data),
                skew(ch_data)
            ])
        
        return np.array(moments)
```

#### 2.4 Hybrid Similarity Scoring

```python
class HybridSimilarityScorer:
    """
    Combines multiple similarity signals into final score
    """
    
    def __init__(self):
        # Configurable weights for different components
        self.weights = {
            'perceptual_hash': 0.25,
            'color_histogram': 0.20,
            'texture_features': 0.20,
            'feature_matching': 0.20,
            'dominant_colors': 0.10,
            'metadata': 0.05
        }
    
    def compute_similarity(self, query_features: dict,
                          candidate_features: dict,
                          metadata_sim: float = 0.0) -> dict:
        """
        Compute comprehensive similarity score
        """
        scores = {}
        
        # 1. Perceptual hash similarity
        scores['perceptual_hash'] = self._hash_similarity(
            query_features['hashes'],
            candidate_features['hashes']
        )
        
        # 2. Color histogram similarity
        scores['color_histogram'] = self._histogram_similarity(
            query_features['color']['lab_histogram'],
            candidate_features['color']['lab_histogram']
        )
        
        # 3. Texture feature similarity
        scores['texture_features'] = self._cosine_similarity(
            query_features['texture'],
            candidate_features['texture']
        )
        
        # 4. Feature matching score
        scores['feature_matching'] = self._feature_match_score(
            query_features['keypoints'],
            candidate_features['keypoints']
        )
        
        # 5. Dominant color similarity
        scores['dominant_colors'] = self._dominant_color_similarity(
            query_features['color']['dominant_colors'],
            candidate_features['color']['dominant_colors']
        )
        
        # 6. Metadata similarity
        scores['metadata'] = metadata_sim
        
        # Compute weighted final score
        final_score = sum(
            self.weights[key] * scores[key]
            for key in self.weights.keys()
        )
        
        return {
            'final_score': final_score,
            'component_scores': scores
        }
    
    def _hash_similarity(self, hash1: dict, hash2: dict) -> float:
        """
        Multi-hash similarity (average of all hash types)
        """
        similarities = []
        for hash_type in ['phash', 'dhash', 'whash']:
            if hash_type in hash1 and hash_type in hash2:
                hamming_dist = self._hamming_distance(
                    hash1[hash_type],
                    hash2[hash_type]
                )
                # Convert to similarity (0-1 scale)
                similarity = 1.0 - (hamming_dist / 256.0)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _histogram_similarity(self, hist1: np.ndarray, 
                             hist2: np.ndarray) -> float:
        """
        Compare histograms using correlation method
        """
        correlation = cv2.compareHist(
            hist1.astype(np.float32),
            hist2.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        # Correlation ranges from -1 to 1, normalize to 0-1
        return (correlation + 1.0) / 2.0
    
    def _cosine_similarity(self, vec1: np.ndarray, 
                          vec2: np.ndarray) -> float:
        """
        Cosine similarity between feature vectors
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _dominant_color_similarity(self, colors1: np.ndarray,
                                   colors2: np.ndarray) -> float:
        """
        Compare dominant color palettes
        Uses Hungarian algorithm for optimal color matching
        """
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        
        # Compute pairwise distances between all colors
        distances = cdist(colors1, colors2, metric='euclidean')
        
        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(distances)
        
        # Average distance of optimal matching
        avg_distance = distances[row_ind, col_ind].mean()
        
        # Convert distance to similarity (normalize by max possible distance)
        max_distance = np.sqrt(3 * 255**2)  # Max RGB distance
        similarity = 1.0 - (avg_distance / max_distance)
        
        return max(0.0, similarity)
```

#### 2.5 Database Schema (PostgreSQL)

```sql
-- Main fabric catalog table
CREATE TABLE fabric_catalog (
    id SERIAL PRIMARY KEY,
    design_no VARCHAR(50) UNIQUE NOT NULL,
    source_pdf VARCHAR(255),
    width_inches DECIMAL(5,2),
    stock INTEGER,
    gsm INTEGER,
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index table for search features
CREATE TABLE fabric_search_index (
    id SERIAL PRIMARY KEY,
    fabric_id INTEGER REFERENCES fabric_catalog(id) ON DELETE CASCADE,
    
    -- Perceptual hashes
    phash_16 VARCHAR(64),
    dhash_16 VARCHAR(64),
    ahash_16 VARCHAR(64),
    whash_16 VARCHAR(64),
    
    -- Color features (stored as vectors using pgvector extension)
    lab_histogram vector(512),
    hsv_histogram vector(128),
    dominant_colors vector(15),  -- 5 colors * 3 channels
    color_moments vector(9),
    
    -- Texture features
    texture_features vector(256),
    
    -- Keypoint descriptors (simplified, top 100 keypoints)
    orb_descriptors bytea,
    
    -- Metadata for quick filtering
    has_stripes BOOLEAN,
    primary_color VARCHAR(20),
    pattern_type VARCHAR(50),
    
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(fabric_id)
);

-- Indexes for fast lookups
CREATE INDEX idx_phash ON fabric_search_index(phash_16);
CREATE INDEX idx_dhash ON fabric_search_index(dhash_16);
CREATE INDEX idx_gsm_range ON fabric_catalog(gsm);
CREATE INDEX idx_stock ON fabric_catalog(stock);

-- Vector similarity indexes (using pgvector)
CREATE INDEX idx_lab_histogram_vector 
    ON fabric_search_index 
    USING ivfflat (lab_histogram vector_cosine_ops);

CREATE INDEX idx_texture_features_vector 
    ON fabric_search_index 
    USING ivfflat (texture_features vector_cosine_ops);

-- Search query tracking
CREATE TABLE search_queries (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(50) UNIQUE,
    query_image_hash VARCHAR(64),
    top_k INTEGER,
    min_similarity DECIMAL(3,2),
    filters JSONB,
    results_count INTEGER,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Search results for analytics
CREATE TABLE search_results (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(50) REFERENCES search_queries(query_id),
    fabric_id INTEGER REFERENCES fabric_catalog(id),
    similarity_score DECIMAL(4,3),
    rank INTEGER,
    clicked BOOLEAN DEFAULT FALSE
);
```

#### 2.6 Caching Strategy (Redis)

```python
class SearchCache:
    """
    Redis-based caching for search results and features
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour
    
    def cache_query_result(self, query_hash: str, 
                          results: list, ttl: int = None):
        """
        Cache search results by query image hash
        """
        cache_key = f"search:result:{query_hash}"
        self.redis.setex(
            cache_key,
            ttl or self.cache_ttl,
            json.dumps(results)
        )
    
    def get_cached_result(self, query_hash: str) -> Optional[list]:
        """
        Retrieve cached search results
        """
        cache_key = f"search:result:{query_hash}"
        cached = self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def cache_fabric_features(self, fabric_id: int, features: dict):
        """
        Cache extracted features for a fabric
        Useful when processing multiple queries
        """
        cache_key = f"fabric:features:{fabric_id}"
        
        # Serialize numpy arrays
        serialized = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in features.items()
        }
        
        self.redis.setex(
            cache_key,
            self.cache_ttl * 24,  # 24 hours for feature cache
            json.dumps(serialized)
        )
    
    def get_cached_features(self, fabric_id: int) -> Optional[dict]:
        """
        Retrieve cached fabric features
        """
        cache_key = f"fabric:features:{fabric_id}"
        cached = self.redis.get(cache_key)
        
        if cached:
            data = json.loads(cached)
            # Deserialize arrays
            return {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in data.items()
            }
        return None
```

#### 2.7 Enhanced API Endpoints

**POST /api/v2/search/image** (Enhanced version)

Additional features:
- Configurable similarity algorithm weights
- Advanced filtering options
- Debug mode with detailed scores

**POST /api/v2/index/batch**

Batch index new images:
```json
{
  "images": [
    {
      "design_no": "JST/20",
      "image_path": "/data/images/new_design.jpg",
      "metadata": {
        "width": "60\"",
        "stock": 500,
        "gsm": 120
      }
    }
  ]
}
```

**GET /api/v2/analytics/search-stats**

Return search analytics:
```json
{
  "total_searches": 1523,
  "avg_processing_time_ms": 432,
  "most_searched_fabrics": [
    {"design_no": "JST/15", "search_count": 87},
    {"design_no": "JCP/64", "search_count": 64}
  ],
  "popular_filters": {
    "gsm_range": {"min": 100, "max": 250},
    "stock_available": 0.82
  }
}
```

**GET /api/v2/similar/{design_no}**

Find similar items to existing catalog item:
```json
{
  "design_no": "JST/15",
  "similar_items": [
    {
      "design_no": "JST/12",
      "similarity": 0.88,
      "similar_because": ["color_match", "stripe_pattern"]
    }
  ]
}
```

### Phase 2 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query Latency (P50) | <1.5s | 50th percentile response time |
| Query Latency (P95) | <2.5s | 95th percentile response time |
| Indexing Speed | >200 images/min | Batch indexing throughput |
| Accuracy (Top-5) | >85% | Correct match in top 5 |
| Cache Hit Rate | >60% | Percentage of cached queries |
| Concurrent Users | 50+ | Simultaneous search requests |
| Database Size | <500MB | For 10k fabric index |

### Phase 2 Deliverables

1. **Production API Service**
   - Multi-worker Uvicorn deployment
   - Nginx reverse proxy config
   - SSL/TLS certificates
   - Rate limiting

2. **Database Layer**
   - PostgreSQL with pgvector
   - Automated backup scripts
   - Migration management

3. **Monitoring & Logging**
   - Prometheus metrics
   - Grafana dashboards
   - Structured logging (JSON)
   - Error tracking (Sentry)

4. **Admin Tools**
   - Bulk index management CLI
   - Search quality evaluation tools
   - A/B testing framework

5. **Documentation**
   - Complete API reference
   - Deployment guide
   - Tuning guide for accuracy/speed tradeoffs

---

## Phase 3: Advanced ML System

### Timeline: Future (2-4 weeks after Phase 2)

### Objectives

Achieve state-of-the-art accuracy (95%+) using deep learning, handle complex scenarios like partial matches and style variations, and enable customer-provided images.

### Machine Learning Approach

#### 3.1 Siamese Network Architecture

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class FabricSiameseNetwork:
    """
    Custom Siamese network for fabric similarity learning
    """
    
    def build_encoder(self, input_shape=(512, 512, 3)):
        """
        Shared encoder network based on EfficientNetB0
        """
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        
        # Fine-tune last 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        inputs = layers.Input(shape=input_shape)
        x = base_model(inputs)
        
        # Add custom embedding layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Final embedding (L2 normalized)
        embeddings = layers.Dense(128)(x)
        embeddings = layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1)
        )(embeddings)
        
        return Model(inputs, embeddings, name='encoder')
    
    def build_siamese_model(self):
        """
        Complete Siamese network with triplet loss
        """
        encoder = self.build_encoder()
        
        # Three inputs: anchor, positive, negative
        anchor_input = layers.Input(shape=(512, 512, 3), 
                                     name='anchor')
        positive_input = layers.Input(shape=(512, 512, 3), 
                                       name='positive')
        negative_input = layers.Input(shape=(512, 512, 3), 
                                       name='negative')
        
        # Generate embeddings
        anchor_embedding = encoder(anchor_input)
        positive_embedding = encoder(positive_input)
        negative_embedding = encoder(negative_input)
        
        # Create model
        model = Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=[anchor_embedding, positive_embedding, 
                    negative_embedding]
        )
        
        return model, encoder
```

#### 3.2 Training Data Generation

```python
class FabricTripletDataGenerator:
    """
    Generate triplet training data from fabric catalog
    """
    
    def __init__(self, catalog_df, augmentation=True):
        self.catalog = catalog_df
        self.augmentation = augmentation
        self.groups = catalog_df.groupby('pattern_type')
    
    def generate_triplet(self):
        """
        Generate (anchor, positive, negative) triplet
        """
        # 1. Select anchor
        anchor_row = self.catalog.sample(1).iloc[0]
        anchor_img = self.load_and_preprocess(anchor_row['image_path'])
        
        # 2. Select positive (same design or similar pattern)
        positive_candidates = self.catalog[
            (self.catalog['pattern_type'] == anchor_row['pattern_type']) &
            (self.catalog['design_no'] != anchor_row['design_no'])
        ]
        
        if len(positive_candidates) > 0:
            positive_row = positive_candidates.sample(1).iloc[0]
        else:
            # Same image with augmentation
            positive_row = anchor_row
        
        positive_img = self.load_and_preprocess(
            positive_row['image_path'],
            augment=True
        )
        
        # 3. Select hard negative (different pattern, but similar color)
        hard_negative_candidates = self.catalog[
            (self.catalog['pattern_type'] != anchor_row['pattern_type']) &
            (self.catalog['primary_color'] == anchor_row['primary_color'])
        ]
        
        if len(hard_negative_candidates) > 0:
            negative_row = hard_negative_candidates.sample(1).iloc[0]
        else:
            # Any different pattern
            negative_row = self.catalog[
                self.catalog['pattern_type'] != anchor_row['pattern_type']
            ].sample(1).iloc[0]
        
        negative_img = self.load_and_preprocess(negative_row['image_path'])
        
        return anchor_img, positive_img, negative_img
    
    def load_and_preprocess(self, image_path, augment=False):
        """
        Load image with optional augmentation
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        
        if augment and self.augmentation:
            img = self.apply_augmentation(img)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        return img
    
    def apply_augmentation(self, img):
        """
        Random augmentation pipeline
        """
        import albumentations as A
        
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.GaussianBlur(p=0.2),
            ], p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.5
            )
        ])
        
        augmented = transform(image=img)
        return augmented['image']
```

#### 3.3 Training Pipeline

```python
class FabricSimilarityTrainer:
    """
    Training orchestration for Siamese network
    """
    
    def __init__(self, model, data_generator):
        self.model = model
        self.data_generator = data_generator
        self.triplet_loss = self.create_triplet_loss()
    
    def create_triplet_loss(self, margin=0.5):
        """
        Triplet loss function
        """
        def triplet_loss(y_true, y_pred):
            # y_pred contains [anchor, positive, negative] embeddings
            anchor = y_pred[0]
            positive = y_pred[1]
            negative = y_pred[2]
            
            # Compute distances
            pos_dist = tf.reduce_sum(
                tf.square(anchor - positive), axis=-1
            )
            neg_dist = tf.reduce_sum(
                tf.square(anchor - negative), axis=-1
            )
            
            # Triplet loss: max(pos_dist - neg_dist + margin, 0)
            loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
            
            return tf.reduce_mean(loss)
        
        return triplet_loss
    
    def train(self, epochs=50, batch_size=32):
        """
        Training loop with hard negative mining
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=self.triplet_loss
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train
        history = self.model.fit(
            self.create_dataset(batch_size),
            epochs=epochs,
            validation_data=self.create_validation_dataset(),
            callbacks=callbacks
        )
        
        return history
```

### Phase 3 Integration

The trained encoder will be deployed alongside the Phase 2 system:

```python
class MLEnhancedSearch:
    """
    Hybrid search combining Phase 2 features with ML embeddings
    """
    
    def __init__(self, encoder_model_path):
        self.encoder = tf.keras.models.load_model(encoder_model_path)
        self.phase2_scorer = HybridSimilarityScorer()
    
    def search(self, query_image, top_k=10):
        """
        Two-stage search:
        1. Phase 2 filtering (fast, broad)
        2. ML re-ranking (accurate, focused)
        """
        # Stage 1: Get top 100 candidates using Phase 2
        candidates = self.phase2_scorer.get_candidates(
            query_image, 
            top_k=100
        )
        
        # Stage 2: Re-rank using ML embeddings
        query_embedding = self.encoder.predict(
            np.expand_dims(query_image, axis=0)
        )[0]
        
        ml_scores = []
        for candidate in candidates:
            candidate_embedding = candidate['ml_embedding']
            
            # Cosine similarity in embedding space
            similarity = np.dot(query_embedding, candidate_embedding)
            ml_scores.append(similarity)
        
        # Combine Phase 2 and ML scores
        final_scores = [
            0.4 * candidates[i]['phase2_score'] + 0.6 * ml_scores[i]
            for i in range(len(candidates))
        ]
        
        # Sort and return top-K
        ranked_indices = np.argsort(final_scores)[::-1][:top_k]
        
        return [candidates[i] for i in ranked_indices]
```

---

## API Specifications

### Base URL
```
Production: https://api.fabricsearch.com/v2
Staging: https://staging-api.fabricsearch.com/v2
```

### Authentication
```
Header: X-API-Key: <your_api_key>
Rate Limit: 100 requests/minute per key
```

### Complete Endpoint List

#### Search Endpoints

**POST /search/image**
Upload image and search for similar fabrics

**POST /search/batch**
Search multiple images in one request (max 10)

**GET /search/history**
Retrieve user's search history

#### Index Management

**POST /index/add**
Add new fabric to search index

**POST /index/batch**
Bulk index multiple fabrics

**DELETE /index/{design_no}**
Remove fabric from index

**PUT /index/{design_no}/update**
Update fabric metadata and re-index

#### Analytics

**GET /analytics/search-stats**
Search usage statistics

**GET /analytics/popular-fabrics**
Most searched fabric designs

**GET /analytics/similarity-matrix**
Generate similarity matrix for catalog analysis

#### Catalog

**GET /catalog/fabrics**
List all fabrics with pagination

**GET /catalog/fabric/{design_no}**
Get details for specific fabric

**GET /similar/{design_no}**
Find similar fabrics to a catalog item

---

## Data Models

### Fabric Catalog Entry
```python
class FabricCatalog(BaseModel):
    design_no: str
    source_pdf: Optional[str]
    width_inches: Optional[float]
    stock: int
    gsm: int
    image_url: str
    pattern_type: Optional[str]
    primary_color: Optional[str]
    has_stripes: bool
    created_at: datetime
    updated_at: datetime
```

### Search Request
```python
class SearchRequest(BaseModel):
    image: str  # base64 or URL
    top_k: int = 10
    min_similarity: float = 0.75
    filters: Optional[SearchFilters] = None
    return_debug_info: bool = False
    
class SearchFilters(BaseModel):
    gsm_min: Optional[int]
    gsm_max: Optional[int]
    width_min: Optional[float]
    width_max: Optional[float]
    stock_available: bool = False
    pattern_types: Optional[List[str]]
    colors: Optional[List[str]]
```

### Search Response
```python
class SearchResult(BaseModel):
    design_no: str
    similarity_score: float
    image_url: str
    metadata: FabricMetadata
    match_details: Optional[MatchDetails]
    rank: int

class SearchResponse(BaseModel):
    query_id: str
    results: List[SearchResult]
    total_found: int
    processing_time_ms: int
    cached: bool
    debug_info: Optional[DebugInfo]
```

---

## Infrastructure Requirements

### Development Environment
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+ with pgvector
- Redis 7.0+
- 16GB RAM minimum
- GPU optional (for Phase 3)

### Production Environment (Phase 2)
- **Application Servers**: 2x 8-core CPU, 16GB RAM
- **Database**: PostgreSQL 15 (4-core, 16GB RAM, 500GB SSD)
- **Cache**: Redis (2-core, 8GB RAM)
- **Storage**: 1TB SSD for images
- **Load Balancer**: Nginx or AWS ALB
- **CDN**: CloudFlare or AWS CloudFront for image delivery

### Scaling Estimates
- 10,000 fabric catalog: ~500MB index, ~10GB images
- 100 concurrent users: 2 app servers
- 1,000 searches/day: Current infrastructure sufficient
- 10,000 searches/day: Add 2 more app servers, read replicas

---

## Testing Strategy

### Unit Tests
- Image preprocessing functions
- Hash generation accuracy
- Similarity scoring algorithms
- Database operations

### Integration Tests
- Full search pipeline
- Index building process
- API endpoint validation
- Cache behavior

### Performance Tests
- Load testing (100 concurrent users)
- Stress testing (sustained high load)
- Latency benchmarks
- Database query optimization

### Accuracy Testing
- Ground truth dataset (manually labeled)
- A/B testing framework
- Confusion matrix analysis
- Precision/Recall curves

---

## Deployment Plan

### Phase 1 Deployment (Days 1-2)
1. Docker containerization
2. Local testing
3. Deploy to staging
4. Internal team validation

### Phase 2 Deployment (Week 2-3)
1. Database migration scripts
2. Feature extraction for existing catalog
3. Staging deployment with full dataset
4. Load testing
5. Production deployment (blue-green)
6. Monitor for 48 hours

### Phase 3 Deployment (Future)
1. Model training on cloud GPU
2. Model validation and benchmarking
3. A/B testing (10% traffic)
4. Gradual rollout to 100%

---

## Future Enhancements

1. **Multi-modal Search**: Text + Image queries
2. **Style Transfer**: "Find fabrics that would look good with this design"
3. **Trend Analysis**: Predict trending patterns
4. **Automated Tagging**: ML-based pattern/color classification
5. **Mobile SDK**: Native mobile app integration
6. **Real-time Indexing**: Add fabrics instantly to search
7. **3D Visualization**: Show how fabric appears in different lighting

---

## Appendices

### A. Performance Benchmarks
*To be populated after Phase 1 completion*

### B. Accuracy Test Results
*To be populated with ground truth evaluation*

### C. Cost Analysis
*Infrastructure and operational costs*

### D. Training Data Requirements (Phase 3)
- Minimum 1,000 fabric images
- At least 5 variations per design
- Manual similarity labels for 500 pairs

---

**Document End**

*Total Word Count: 5,200+ words*