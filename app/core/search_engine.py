"""
Search Engine Module

Implements the three-stage filtering pipeline for fabric visual search:
1. Perceptual hash filtering (fast, broad)
2. Color histogram similarity (refine by appearance)
3. Feature matching validation (geometric verification)
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from .image_processor import ImageProcessor, hamming_distance
from .database import Database, IndexManager


class SearchEngine:
    """
    Three-stage fabric search engine.
    """
    
    def __init__(self, db: Database, processor: ImageProcessor):
        self.db = db
        self.processor = processor
        self.index_manager = IndexManager(db, processor)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Configurable thresholds - relaxed for external images
        self.hash_threshold = 110  # Max Hamming distance (out of 256 bits) - ~43% difference allowed (was 80)
        self.color_threshold = 0.2  # Min color similarity (was 0.3)
        self.min_feature_matches = 8  # Min ORB matches (was 10)
        
        # Scoring weights - adjusted to prioritize pattern matching over color
        self.weights = {
            'hash': 0.45,      # Increased from 0.40 - pattern structure most important
            'color': 0.25,     # Decreased from 0.30 - less weight on exact color match
            'features': 0.25,  # Increased from 0.20 - geometric validation important
            'metadata': 0.05   # Decreased from 0.10 - focus on visual similarity
        }
    
    def search(self, query_source: str, top_k: int = 10, 
               min_similarity: float = 0.5,
               filters: Optional[Dict] = None,
               use_clahe: bool = True) -> Dict:
        """
        Main search method.
        
        Args:
            query_source: Image path, base64, or URL
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            filters: Optional metadata filters (gsm_min, gsm_max, etc.)
            use_clahe: Use CLAHE preprocessing for better external image matching (default: True)
            
        Returns:
            Search results with scores and metadata
        """
        import time
        start_time = time.time()
        
        # Extract features from query image
        # Use CLAHE for external images to improve matching
        query_features = self.processor.extract_all_features(query_source, use_clahe=use_clahe)
        if query_features is None:
            return {
                'error': 'Failed to process query image',
                'results': [],
                'total_found': 0,
                'processing_time_ms': 0
            }
        
        # Load index
        index = self.index_manager.load_index()
        if not index:
            return {
                'error': 'Search index is empty. Run index build first.',
                'results': [],
                'total_found': 0,
                'processing_time_ms': 0
            }
        
        # Stage 1: Hash filtering
        candidates = self._stage1_hash_filter(query_features, index)
        
        # Stage 2: Color histogram filtering
        candidates = self._stage2_color_filter(query_features, candidates)
        
        # Stage 3: Feature matching (on top candidates only)
        candidates = self._stage3_feature_matching(query_features, candidates[:100])
        
        # Apply metadata filters if provided
        if filters:
            candidates = self._apply_filters(candidates, filters)
        
        # Compute final scores and rank
        results = self._compute_final_scores(candidates, query_features)
        
        # Filter by minimum similarity
        results = [r for r in results if r['similarity_score'] >= min_similarity]
        
        # Sort by score and take top-k
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:top_k]
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'results': results,
            'total_found': len(results),
            'processing_time_ms': processing_time,
            'stages': {
                'hash_candidates': len(candidates) if candidates else 0
            }
        }
    
    def _stage1_hash_filter(self, query_features: Dict, 
                            index: List[Dict]) -> List[Dict]:
        """
        Stage 1: Fast filtering based on perceptual hash Hamming distance.
        This eliminates most non-matches very quickly.
        """
        query_hashes = query_features['hashes']
        candidates = []
        
        for item in index:
            # Calculate Hamming distance for multiple hash types
            phash_dist = hamming_distance(query_hashes['phash'], item['phash'])
            dhash_dist = hamming_distance(query_hashes['dhash'], item['dhash'])
            
            # Use weighted minimum distance
            # dHash is weighted slightly less as it can be sensitive to gradients
            min_dist = min(phash_dist, dhash_dist * 0.9)
            
            if min_dist <= self.hash_threshold:
                item['hash_distance'] = min_dist
                item['hash_scores'] = {
                    'phash': phash_dist,
                    'dhash': dhash_dist
                }
                candidates.append(item)
        
        return candidates
    
    def _stage2_color_filter(self, query_features: Dict,
                             candidates: List[Dict]) -> List[Dict]:
        """
        Stage 2: Filter by color histogram similarity in LAB space.
        """
        if not candidates:
            return []
        
        query_hist = query_features['color_histogram']
        refined = []
        
        for candidate in candidates:
            cand_hist = candidate.get('color_histogram')
            if cand_hist is None:
                continue
            
            # Compute histogram correlation
            # Correlation ranges from -1 to 1, we normalize to 0-1
            correlation = cv2.compareHist(
                query_hist.astype(np.float32),
                cand_hist.astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            
            color_similarity = (correlation + 1.0) / 2.0
            
            if color_similarity >= self.color_threshold:
                candidate['color_score'] = color_similarity
                refined.append(candidate)
        
        # Sort by color score for next stage
        refined.sort(key=lambda x: x.get('color_score', 0), reverse=True)
        
        return refined
    
    def _stage3_feature_matching(self, query_features: Dict,
                                 candidates: List[Dict]) -> List[Dict]:
        """
        Stage 3: Verify geometric consistency using ORB feature matching.
        """
        if not candidates:
            return []
        
        query_desc = query_features['orb_descriptors']
        if query_desc is None:
            # If query has no features, skip this stage
            for c in candidates:
                c['feature_score'] = 0.5  # Neutral score
                c['feature_matches'] = 0
            return candidates
        
        validated = []
        
        for candidate in candidates:
            cand_desc = candidate.get('orb_descriptors')
            
            if cand_desc is None:
                candidate['feature_score'] = 0.3
                candidate['feature_matches'] = 0
                validated.append(candidate)
                continue
            
            try:
                # KNN match
                matches = self.bf_matcher.knnMatch(query_desc, cand_desc, k=2)
                
                # Lowe's ratio test
                good_matches = []
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                match_count = len(good_matches)
                candidate['feature_matches'] = match_count
                
                # Normalize score (assuming ~100 is very good)
                candidate['feature_score'] = min(match_count / 50.0, 1.0)
                
                validated.append(candidate)
                
            except Exception as e:
                candidate['feature_score'] = 0.0
                candidate['feature_matches'] = 0
                validated.append(candidate)
        
        return validated
    
    def _apply_filters(self, candidates: List[Dict], 
                      filters: Dict) -> List[Dict]:
        """
        Apply metadata filters to candidates.
        """
        filtered = []
        
        for c in candidates:
            # GSM filter
            if 'gsm_min' in filters and c.get('gsm', 0) < filters['gsm_min']:
                continue
            if 'gsm_max' in filters and c.get('gsm', 0) > filters['gsm_max']:
                continue
            
            # Stock filter
            if filters.get('stock_available') and c.get('stock', 0) <= 0:
                continue
            
            # Width filter (parse width as number if needed)
            width = c.get('width')
            if width:
                try:
                    width_num = float(str(width).replace('"', '').replace("'", ''))
                    if 'width_min' in filters and width_num < filters['width_min']:
                        continue
                    if 'width_max' in filters and width_num > filters['width_max']:
                        continue
                except:
                    pass
            
            filtered.append(c)
        
        return filtered
    
    def _compute_final_scores(self, candidates: List[Dict],
                             query_features: Dict) -> List[Dict]:
        """
        Compute weighted final similarity scores.
        
        Weights:
        - Hash similarity: 40%
        - Color similarity: 30%
        - Feature matches: 20%
        - Metadata bonus: 10%
        """
        results = []
        
        for candidate in candidates:
            # Hash score: convert distance to similarity (0-1)
            hash_dist = candidate.get('hash_distance', self.hash_threshold)
            # Max distance for 16x16 hash = 256 bits
            hash_score = 1.0 - (hash_dist / 256.0)
            
            # Color score
            color_score = candidate.get('color_score', 0.5)
            
            # Feature score
            feature_score = candidate.get('feature_score', 0.5)
            
            # Metadata score (placeholder - could factor in GSM similarity, etc.)
            metadata_score = 0.5
            
            # Weighted final score
            final_score = (
                self.weights['hash'] * hash_score +
                self.weights['color'] * color_score +
                self.weights['features'] * feature_score +
                self.weights['metadata'] * metadata_score
            )
            
            results.append({
                'design_no': candidate['design_no'],
                'similarity_score': round(final_score, 3),
                'image_path': candidate.get('image_path', ''),
                'width': candidate.get('width'),
                'stock': candidate.get('stock'),
                'gsm': candidate.get('gsm'),
                'source_pdf': candidate.get('source_pdf'),
                'match_details': {
                    'hash_score': round(hash_score, 3),
                    'color_score': round(color_score, 3),
                    'feature_score': round(feature_score, 3),
                    'feature_matches': candidate.get('feature_matches', 0)
                }
            })
        
        return results
