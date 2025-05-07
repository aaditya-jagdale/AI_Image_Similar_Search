import os
import torch
import faiss
from PIL import Image
import numpy as np
import json
from flask import jsonify, current_app, url_for
import glob
from .handle_build_index import model, processor, device, transform_list, INDEX_DIR, INDEX_FILENAME, METADATA_FILENAME 

DEFAULT_K = 10

def preprocess_query_image(image_file_storage):
    """Preprocesses the uploaded image and generates its averaged embedding."""
    try:
        img = Image.open(image_file_storage).convert("RGB")
        variants = [t(img) for t in transform_list]
    except Exception as e:
        print(f"Error processing query image: {e}")
        return None

    if not variants:
        return None

    inputs = processor(images=variants, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    # Average variants
    avg_embedding = features.mean(dim=0).cpu().numpy()
    return avg_embedding.astype('float32').reshape(1, -1) # Reshape for Faiss search

def handle_image_search(image_file_storage, k=DEFAULT_K):
    """Handles the image search request."""
    index_filepath = os.path.join(INDEX_DIR, INDEX_FILENAME)
    metadata_filepath = os.path.join(INDEX_DIR, METADATA_FILENAME)

    # Check if index and metadata files exist
    if not os.path.exists(index_filepath) or not os.path.exists(metadata_filepath):
         return jsonify({"error": "Index files not found. Please build the index first."}), 404

    try:
        index = faiss.read_index(index_filepath)
        with open(metadata_filepath, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading index/metadata: {e}")
        return jsonify({"error": f"Failed to load index or metadata: {e}"}), 500

    query_embedding = preprocess_query_image(image_file_storage)
    if query_embedding is None:
        return jsonify({"error": "Failed to process query image."}), 400

    try:
        # Search for more results than needed to ensure we have enough valid ones
        distances, indices = index.search(query_embedding, min(40, len(metadata)))
    except Exception as e:
        print(f"Error searching index: {e}")
        return jsonify({"error": f"Failed during index search: {e}"}), 500

    results = []
    if indices.size > 0:
        for i, idx in enumerate(indices[0]):
            if idx != -1 and 0 <= idx < len(metadata):
                item = metadata[idx]
                item['distance'] = float(distances[0][i])
                results.append(item)

                if len(results) >= 20:
                    break

    results.sort(key=lambda x: x['distance'])
    results = results[:20]

    return jsonify(results), 200