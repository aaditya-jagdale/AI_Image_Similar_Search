#!/usr/bin/env python3
"""
Test script for external image search improvements
"""
import base64
import requests
import json
import sys

def test_external_image(image_path='test.jpg', use_clahe=True):
    """Test external image search with CLAHE preprocessing"""
    
    # Read and encode image
    print(f"Reading {image_path}...")
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare request
    url = 'http://localhost:8000/api/v1/search'
    payload = {
        'image': image_data,
        'top_k': 10,
        'min_similarity': 0.3,  # Lower threshold to see more results
        'use_clahe': use_clahe
    }
    
    print(f"Searching with use_clahe={use_clahe}, min_similarity=0.3...")
    
    # Make request
    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    
    # Display results
    print(f"\n{'='*70}")
    print(f"Results: {result['total_found']} matches found")
    print(f"Processing time: {result['processing_time_ms']}ms")
    print(f"{'='*70}\n")
    
    if result['total_found'] > 0:
        print("Top 5 matches:\n")
        for i, match in enumerate(result['results'][:5], 1):
            details = match['match_details']
            print(f"{i}. {match['design_no']}")
            print(f"   Overall Score: {match['similarity_score']:.3f}")
            print(f"   Hash Score: {details['hash_score']:.3f}")
            print(f"   Color Score: {details['color_score']:.3f}")
            print(f"   Feature Score: {details['feature_score']:.3f}")
            print(f"   Feature Matches: {details['feature_matches']}")
            print(f"   GSM: {match.get('gsm', 'N/A')}, Width: {match.get('width', 'N/A')}")
            print()
    else:
        print("No matches found. Try lowering min_similarity further.")
    
    return result

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
    
    print("="*70)
    print("TESTING: External Image Search with Improvements")
    print("="*70)
    
    # Test with CLAHE enabled (default)
    print("\n[TEST 1] WITH CLAHE preprocessing (enhanced for external images)")
    result_clahe = test_external_image(image_path, use_clahe=True)
    
    # Test without CLAHE for comparison  
    print("\n[TEST 2] WITHOUT CLAHE preprocessing (standard)")
    result_standard = test_external_image(image_path, use_clahe=False)
    
    # Compare results
    if result_clahe and result_standard:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"With CLAHE:    {result_clahe['total_found']} matches")
        print(f"Without CLAHE: {result_standard['total_found']} matches")
        
        if result_clahe['total_found'] > 0 and result_standard['total_found'] > 0:
            improvement = ((result_clahe['total_found'] - result_standard['total_found']) / 
                          max(result_standard['total_found'], 1) * 100)
            print(f"Improvement: {improvement:+.1f}%")
            
            clahe_top_score = result_clahe['results'][0]['similarity_score']
            std_top_score = result_standard['results'][0]['similarity_score']
            print(f"\nTop match score:")
            print(f"  With CLAHE:    {clahe_top_score:.3f}")
            print(f"  Without CLAHE: {std_top_score:.3f}")
            print(f"  Difference: {clahe_top_score - std_top_score:+.3f}")
