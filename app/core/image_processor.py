"""
Image Processor Module

Handles image preprocessing and feature extraction for fabric visual search.
Implements the Phase 1 pipeline: preprocessing, multi-hash generation,
color histogram computation, and ORB feature extraction.
"""

import cv2
import numpy as np
from PIL import Image
import imagehash
from typing import Dict, Tuple, Optional
import base64
import requests
from io import BytesIO


class ImageProcessor:
    """
    Core image processing for fabric visual search.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256), hash_size: int = 16):
        self.target_size = target_size
        self.hash_size = hash_size
        self.hash_sizes = [8, 16, 24]  # Multi-scale hashing for robustness
        self.orb = cv2.ORB_create(nfeatures=1000)  # Increased from 500 for better feature detection
        
        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def load_image(self, source: str) -> Optional[np.ndarray]:
        """
        Load image from file path, base64 string, or URL.
        
        Args:
            source: File path, base64 string, or URL
            
        Returns:
            BGR image as numpy array, or None if loading fails
        """
        try:
            # Check if it's a base64 string
            if source.startswith('data:image') or self._is_base64(source):
                return self._load_from_base64(source)
            
            # Check if it's a URL
            if source.startswith('http://') or source.startswith('https://'):
                return self._load_from_url(source)
            
            # Otherwise treat as file path
            img = cv2.imread(source)
            if img is None:
                raise ValueError(f"Could not load image from path: {source}")
            return img
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _is_base64(self, s: str) -> bool:
        """Check if string looks like base64 encoded data."""
        try:
            # Base64 strings are typically long and contain only valid base64 chars
            # If it's very long (>1000 chars), it's likely base64
            if len(s) > 1000:
                # Try to decode - if it works, it's valid base64
                base64.b64decode(s, validate=True)
                return True
            # For shorter strings, check if it looks like a file path
            elif len(s) > 100 and not (s.startswith('./') or s.startswith('../') or 
                                        (s.startswith('/') and '.' in s.split('/')[-1][:10])):
                base64.b64decode(s, validate=True)
                return True
        except:
            pass
        return False
    
    def _load_from_base64(self, data: str) -> np.ndarray:
        """Load image from base64 string."""
        # Remove data URL prefix if present
        if ',' in data:
            data = data.split(',')[1]
        
        img_bytes = base64.b64decode(data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode base64 image")
        return img
    
    def _load_from_url(self, url: str) -> np.ndarray:
        """Load image from URL."""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"Could not decode image from URL: {url}")
        return img
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for feature extraction.
        
        Steps:
        1. Convert to RGB
        2. Resize to target size
        3. Apply gray world white balance
        4. Slight Gaussian blur for denoising
        
        Args:
            img: BGR image
            
        Returns:
            Preprocessed RGB image
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        img_resized = cv2.resize(
            img_rgb, 
            self.target_size, 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # White balance correction (gray world assumption)
        img_balanced = self._apply_gray_world_wb(img_resized)
        
        # Slight denoise
        img_denoised = cv2.GaussianBlur(img_balanced, (3, 3), 0)
        
        return img_denoised
    
    def preprocess_enhanced(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate multiple preprocessed representations for robustness.
        This helps match external images with variable lighting/quality.
        
        Args:
            img: BGR image
            
        Returns:
            Dictionary with multiple preprocessed versions:
            - 'standard': Standard preprocessing
            - 'clahe': CLAHE-enhanced for lighting variations
            - 'lab_normalized': LAB space normalized
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        img_resized = cv2.resize(
            img_rgb, 
            self.target_size, 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Standard preprocessing
        img_balanced = self._apply_gray_world_wb(img_resized)
        img_standard = cv2.GaussianBlur(img_balanced, (3, 3), 0)
        
        # CLAHE-enhanced version for better lighting normalization
        img_clahe = self._apply_clahe_normalization(img_resized)
        
        # LAB-normalized version
        img_lab_norm = self._apply_lab_normalization(img_resized)
        
        return {
            'standard': img_standard,
            'clahe': img_clahe,
            'lab_normalized': img_lab_norm
        }
    
    def _apply_clahe_normalization(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space.
        This dramatically improves matching for images with poor/variable lighting.
        
        Args:
            img: RGB image
            
        Returns:
            CLAHE-normalized RGB image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel (luminance)
        l_clahe = self.clahe.apply(l)
        
        # Merge back and convert to RGB
        lab_clahe = cv2.merge([l_clahe, a, b])
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        # Denoise
        img_denoised = cv2.GaussianBlur(img_clahe, (3, 3), 0)
        
        return img_denoised
    
    def _apply_lab_normalization(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image in LAB color space to reduce lighting variations.
        
        Args:
            img: RGB image
            
        Returns:
            LAB-normalized RGB image
        """
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Normalize L channel to full range
        l_min, l_max = l.min(), l.max()
        if l_max > l_min:
            l_normalized = ((l - l_min) / (l_max - l_min) * 255).astype(np.uint8)
        else:
            l_normalized = l
        
        # Merge and convert back
        lab_normalized = cv2.merge([l_normalized, a, b])
        img_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)
        
        # Apply gray world white balance
        img_balanced = self._apply_gray_world_wb(img_normalized)
        
        # Denoise
        img_denoised = cv2.GaussianBlur(img_balanced, (3, 3), 0)
        
        return img_denoised
    
    def _apply_gray_world_wb(self, img: np.ndarray) -> np.ndarray:
        """
        Apply gray world white balance.
        Assumes average color should be gray.
        """
        img_float = img.astype(np.float32)
        avg_channels = img_float.mean(axis=(0, 1))
        avg_gray = avg_channels.mean()
        
        # Avoid division by zero
        scale = np.where(avg_channels > 0, avg_gray / avg_channels, 1.0)
        
        result = img_float * scale
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def generate_hashes(self, img: np.ndarray) -> Dict[str, str]:
        """
        Generate multiple perceptual hashes for robustness.
        
        Uses 16x16 hash size for better precision with detailed fabric patterns.
        
        Args:
            img: RGB image (preprocessed)
            
        Returns:
            Dictionary with hash types and their hex values
        """
        pil_img = Image.fromarray(img)
        
        return {
            'phash': str(imagehash.phash(pil_img, hash_size=self.hash_size)),
            'dhash': str(imagehash.dhash(pil_img, hash_size=self.hash_size)),
            'ahash': str(imagehash.average_hash(pil_img, hash_size=self.hash_size)),
            'whash': str(imagehash.whash(pil_img, hash_size=self.hash_size))
        }
    
    def generate_multiscale_hashes(self, img: np.ndarray) -> Dict[str, Dict[int, str]]:
        """
        Generate perceptual hashes at multiple scales for better robustness.
        This helps match images at different zoom levels or resolutions.
        
        Args:
            img: RGB image (preprocessed)
            
        Returns:
            Dictionary with hash types, each containing multiple scale sizes
        """
        pil_img = Image.fromarray(img)
        
        multiscale = {
            'phash': {},
            'dhash': {},
            'ahash': {},
            'whash': {}
        }
        
        for size in self.hash_sizes:
            multiscale['phash'][size] = str(imagehash.phash(pil_img, hash_size=size))
            multiscale['dhash'][size] = str(imagehash.dhash(pil_img, hash_size=size))
            multiscale['ahash'][size] = str(imagehash.average_hash(pil_img, hash_size=size))
            multiscale['whash'][size] = str(imagehash.whash(pil_img, hash_size=size))
        
        return multiscale
    
    def compute_color_histogram(self, img: np.ndarray) -> np.ndarray:
        """
        Compute color histogram in LAB color space.
        LAB is perceptually uniform - better for matching.
        
        Args:
            img: RGB image
            
        Returns:
            Flattened normalized histogram
        """
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Compute histogram for each channel
        hist_l = cv2.calcHist([lab], [0], None, [32], [0, 256])
        hist_a = cv2.calcHist([lab], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [32], [0, 256])
        
        # Concatenate and normalize
        hist = np.concatenate([hist_l, hist_a, hist_b]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        
        return hist
    
    def extract_orb_features(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Extract ORB keypoints and descriptors.
        
        Args:
            img: RGB image
            
        Returns:
            Tuple of (descriptors, keypoint_count)
        """
        # Convert to grayscale for ORB
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None:
            return None, 0
        
        return descriptors, len(keypoints)
    
    def extract_all_features(self, source: str, use_clahe: bool = False) -> Optional[Dict]:
        """
        Extract all features from an image source.
        
        Args:
            source: File path, base64, or URL
            use_clahe: If True, use CLAHE-enhanced preprocessing for better 
                      external image matching (recommended for user-uploaded images)
            
        Returns:
            Dictionary containing all extracted features
        """
        # Load image
        img = self.load_image(source)
        if img is None:
            return None
        
        # Choose preprocessing method
        if use_clahe:
            # Use CLAHE for external images with variable lighting
            preprocessed_variations = self.preprocess_enhanced(img)
            img_processed = preprocessed_variations['clahe']
        else:
            # Standard preprocessing for catalog images
            img_processed = self.preprocess(img)
        
        # Extract all features
        hashes = self.generate_hashes(img_processed)
        color_hist = self.compute_color_histogram(img_processed)
        orb_desc, kp_count = self.extract_orb_features(img_processed)
        
        return {
            'hashes': hashes,
            'color_histogram': color_hist,
            'orb_descriptors': orb_desc,
            'keypoint_count': kp_count,
            'preprocessed_image': img_processed
        }


# Utility functions for hash comparison
def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Calculate Hamming distance between two hex hash strings.
    """
    # Convert hex strings to integers
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    
    # XOR and count set bits
    xor = int1 ^ int2
    return bin(xor).count('1')


def hash_similarity(hash1: str, hash2: str, max_bits: int = 256) -> float:
    """
    Calculate similarity between two hashes (0-1 scale, higher is better).
    """
    dist = hamming_distance(hash1, hash2)
    return 1.0 - (dist / max_bits)
