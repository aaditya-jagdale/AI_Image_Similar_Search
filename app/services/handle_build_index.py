import os
import torch
import faiss
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import numpy as np
import json
from flask import jsonify # Assuming Flask context, kept for compatibility
import requests
from io import BytesIO
# import fitz # PyMuPDF - No longer needed as handle_build_index is removed

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Configuration ---
INDEX_DIR = "faiss_indices"
BATCH_SIZE = 32  # Adjusted for potentially better GPU utilization, tune as needed
INDEX_FILENAME = "search.faiss"
METADATA_FILENAME = "search_metadata.json"

# Create output directory if it doesn't exist
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Model and Processor Initialization ---
# Using a try-except block for model loading can be good practice
try:
    MODEL_NAME = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()  # Set model to evaluation mode
except Exception as e:
    print(f"Error loading CLIP model/processor: {e}")
    # Depending on application, might want to exit or raise
    raise

# --- Device Selection (CUDA > MPS > CPU) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # MPS for Apple Silicon
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device)
print(f"Using device: {device}")


# --- Image Augmentation Transforms ---
# This list defines various transformations applied to each image.
# The goal is to create multiple "views" of an image, and the average
# embedding of these views can be more robust.
# Accessing crop_size for target dimensions.
try:
    target_crop_height = processor.image_processor.crop_size["height"]
except AttributeError:
    print("Warning: processor.image_processor.crop_size not found. Falling back to default 224.")
    # Fallback if crop_size attribute doesn't exist, though it should for CLIPImageProcessor
    target_crop_height = 224 
except KeyError:
    print(f"Warning: 'height' key not in processor.image_processor.crop_size. "
          f"Available keys: {processor.image_processor.crop_size.keys()}. Falling back to default 224.")
    target_crop_height = 224


transform_list = [
    transforms.Compose([]),  # Original image (processor will handle its base transforms)
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]), # Always flip
    transforms.Compose([transforms.RandomRotation(degrees=15)]), # Reduced rotation
    transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)]), # Softer jitter
    # Corrected line: using processor.image_processor.crop_size["height"]
    transforms.Compose([transforms.RandomResizedCrop(target_crop_height, scale=(0.9, 1.0), ratio=(0.9, 1.1))]),
    transforms.Compose([transforms.GaussianBlur(kernel_size=3)]), # Reduced blur
    transforms.Compose([transforms.RandomPerspective(distortion_scale=0.2, p=0.5)]), # Reduced perspective, probabilistic
    transforms.Compose([transforms.Grayscale(num_output_channels=3)]), # Grayscale version
]
print(f"Using {len(transform_list)} augmentations per image. Target crop height: {target_crop_height}")

def preprocess_variants(image_obj: Image.Image):
    """
    Applies a list of transformations to an image to generate variants.
    Args:
        image_obj (PIL.Image.Image): The input image.
    Returns:
        list[PIL.Image.Image]: A list of transformed PIL images. Returns empty if an error occurs.
    """
    if not image_obj:
        return []
    try:
        img_rgb = image_obj.convert("RGB")
        variants = []
        for t_func in transform_list:
            try:
                variants.append(t_func(img_rgb))
            except Exception as e_transform:
                print(f"Warning: Failed to apply a transform: {e_transform}. Skipping this variant.")
        return variants
    except Exception as e:
        print(f"Error processing image for variant generation: {e}")
        return []


def process_batch(batch_image_data: list):
    """
    Processes a batch of image data to generate embeddings.
    Each image in the batch is augmented, and the embeddings of its variants are averaged.

    Args:
        batch_image_data (list): A list of dicts, each containing 'image_obj' (PIL.Image)
                                 and 'metadata'.
    Returns:
        tuple: (list of numpy arrays (embeddings), list of metadata dicts)
               Returns ([], []) if processing fails or no valid embeddings are generated.
    """
    all_variants_pil = []      # Stores all PIL image variants from all original images
    num_variants_per_image = [] # Stores count of variants for each original image that yielded variants
    valid_metadata_for_batch = [] # Metadata for original images that yielded variants

    for item_data in batch_image_data:
        pil_image_obj = item_data.get('image_obj')
        
        current_image_variants = preprocess_variants(pil_image_obj)
        
        if current_image_variants:
            all_variants_pil.extend(current_image_variants)
            num_variants_per_image.append(len(current_image_variants))
            valid_metadata_for_batch.append(item_data["metadata"])
        
        if pil_image_obj:
            try:
                pil_image_obj.close() # Close the PIL image object after processing
            except Exception:
                pass # Ignore errors on close

    if not all_variants_pil:
        print("No processable image variants found in the current batch.")
        return [], []

    all_variants_pil = [img for img in all_variants_pil if isinstance(img, Image.Image)]
    if not all_variants_pil:
        print("No valid PIL images remained after filtering in the current batch.")
        return [], []

    expected_total_variants = sum(num_variants_per_image)
    if len(all_variants_pil) != expected_total_variants:
        print(f"Critical Internal Inconsistency: Number of PIL variants ({len(all_variants_pil)}) "
              f"does not match expected sum from map ({expected_total_variants}). Skipping batch.")
        return [], []

    try:
        inputs = processor(images=all_variants_pil, return_tensors="pt", padding="max_length", truncation=True).to(device)
    except Exception as e:
        print(f"Error during CLIP processing: {e}. Skipping batch.")
        return [], []

    with torch.no_grad():
        try:
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize features
        except Exception as e:
            print(f"Error during model inference (get_image_features): {e}. Skipping batch.")
            return [], []

    if features.shape[0] != len(all_variants_pil):
        print(f"Warning: Mismatch in embedding generation. "
              f"Input PIL images: {len(all_variants_pil)}, Output embeddings: {features.shape[0]}. "
              "Skipping batch.")
        return [], []

    final_embeddings_for_batch = []
    current_feature_idx = 0
    for count_variants_for_one_image in num_variants_per_image:
        if current_feature_idx + count_variants_for_one_image > features.shape[0]:
             print(f"Critical Error: Aggregation indexing out of bounds. "
                  f"Current_idx: {current_feature_idx}, num_variants: {count_variants_for_one_image}, features_shape: {features.shape[0]}. "
                  "Skipping batch.")
             return [], [] 

        image_specific_variant_embeddings = features[current_feature_idx : current_feature_idx + count_variants_for_one_image]
        average_embedding_for_image = image_specific_variant_embeddings.mean(dim=0)
        final_embeddings_for_batch.append(average_embedding_for_image.cpu().numpy())
        current_feature_idx += count_variants_for_one_image
    
    return final_embeddings_for_batch, valid_metadata_for_batch


def handle_build_index_from_json(json_data: list):
    """
    Builds a FAISS index from image URLs specified in the provided JSON data.
    Args:
        json_data (list): A list of objects, where each object contains an "image" URL
                          and other metadata.
    Returns:
        Flask Response, status_code
    """
    if not isinstance(json_data, list):
        return jsonify({"error": "JSON data must be a list of objects"}), 400

    image_data_for_processing = []
    total_items_in_json = len(json_data)
    print(f"Preparing image data from {total_items_in_json} JSON items...")

    for item in tqdm(json_data, desc="Downloading and Preparing Images"):
        image_url = item.get("image")
        if not image_url or not isinstance(image_url, str):
            design_no = item.get('design_no', item.get('id', 'N/A'))
            print(f"Warning: Skipping item {design_no} due to missing or invalid 'image' URL.")
            continue

        try:
            response = requests.get(image_url, stream=True, timeout=60)
            response.raise_for_status() 
            
            content_type = response.headers.get('Content-Type', '').lower()
            if not content_type.startswith('image/'):
                 print(f"Warning: URL {image_url} does not appear to be an image (Content-Type: {content_type}). Skipping.")
                 continue

            image_bytes_content = BytesIO(response.content)
            pil_image_obj = Image.open(image_bytes_content)
            pil_image_obj = pil_image_obj.copy() 

            item_metadata = item.copy() 
            item_metadata["source_type"] = "json_url"
            image_data_for_processing.append({
                "image_obj": pil_image_obj,
                "metadata": item_metadata
            })
        except requests.exceptions.RequestException as req_e:
            print(f"Warning: Failed to download image from {image_url}: {req_e}")
            continue
        except Image.UnidentifiedImageError:
            print(f"Warning: Could not identify image from URL {image_url}. It might be corrupted or not a supported format.")
            continue
        except Exception as img_e: 
            print(f"Warning: Error processing image from {image_url}: {img_e}")
            continue
    
    if not image_data_for_processing:
        return jsonify({"error": "No images could be successfully downloaded or processed from the provided JSON data."}), 400

    all_generated_embeddings = []
    all_valid_metadata_for_index = []
    print(f"Generating embeddings for {len(image_data_for_processing)} downloaded images...")

    for i in tqdm(range(0, len(image_data_for_processing), BATCH_SIZE), desc="Embedding JSON Batches"):
        batch_data_to_process = image_data_for_processing[i:i+BATCH_SIZE]
        embeddings_from_batch, metadata_from_batch = process_batch(batch_data_to_process)
        
        if embeddings_from_batch:
            all_generated_embeddings.extend(embeddings_from_batch)
            all_valid_metadata_for_index.extend(metadata_from_batch)

    del image_data_for_processing
    if device.type == 'cuda':
        torch.cuda.empty_cache()


    if not all_generated_embeddings:
        return jsonify({"error": "No embeddings were generated from the downloaded images. Check image URLs, formats, and logs."}), 500

    try:
        embedding_matrix = np.vstack(all_generated_embeddings).astype('float32')
    except ValueError as ve:
         return jsonify({"error": f"Failed to stack embeddings. Ensure all embeddings have the same dimension. Details: {ve}"}), 500

    if embedding_matrix.size == 0:
         return jsonify({"error": "Embedding matrix is empty after processing all batches."}), 500
    
    expected_dim = model.config.projection_dim
    if embedding_matrix.shape[1] != expected_dim:
        print(f"Error: Embedding dimension mismatch. Expected {expected_dim}, got {embedding_matrix.shape[1]}.")
        return jsonify({"error": f"Embedding dimension mismatch. Expected {expected_dim}, got {embedding_matrix.shape[1]}"}), 500

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    index_filepath = os.path.join(INDEX_DIR, INDEX_FILENAME) 
    metadata_filepath = os.path.join(INDEX_DIR, METADATA_FILENAME) 

    try:
        faiss.write_index(index, index_filepath)
        with open(metadata_filepath, "w") as f:
            json.dump(all_valid_metadata_for_index, f, indent=4)

        print(f"✅ FAISS index saved to '{index_filepath}' with {index.ntotal} items.")
        print(f"✅ Metadata saved to '{metadata_filepath}' for {len(all_valid_metadata_for_index)} items.")

        return jsonify({
            "message": f"Index built successfully from provided JSON data",
            "local_index_path": index_filepath,
            "local_metadata_path": metadata_filepath,
            "num_items_indexed": index.ntotal
        }), 200
    except Exception as e:
        print(f"Error saving FAISS index or metadata: {e}")
        return jsonify({"error": f"Failed saving index/metadata: {e}"}), 500

# Example usage (conceptual, assuming you run this outside Flask or have mock objects)
# if __name__ == '__main__':
#     print("Optimized FAISS indexing script (JSON focus) loaded.")
#     print(f"To build an index, call 'handle_build_index_from_json(json_data)'.")

#     # --- Mock JSON data for testing handle_build_index_from_json ---
#     mock_json_input = [
#         {
#             "design_no": "test_001",
#             "category": "test_cat",
#             "image": "https://placehold.co/600x400/EEE/31343C?text=Test+Image+1"
#         },
#         {
#             "design_no": "test_002",
#             "category": "test_cat",
#             "image": "https://placehold.co/800x600/CCC/31343C?text=Test+Image+2"
#         },
#         { 
#             "design_no": "test_003_invalid_url",
#             "image": "https://thisshouldnotresolve123abcxyz.com/image.jpg"
#         },
#         { 
#             "design_no": "test_004_not_image_url",
#             "image": "https://google.com" # This URL points to HTML, not an image
#         }
#     ]
#     print("\n--- Testing JSON Indexing ---")
#     # In a non-Flask environment, jsonify won't work as expected.
#     # The test call will print messages and if an error occurs in jsonify,
#     # it will raise an exception here (e.g., RuntimeError: Working outside of application context).
#     # For standalone testing, you might return dicts directly from handle_build_index_from_json
#     # or mock the Flask app context.
#     try:
#         response_data, status_code_json = handle_build_index_from_json(mock_json_input)
#         # If running outside Flask, response_data is a Flask Response object.
#         # To see its JSON content:
#         if hasattr(response_data, 'get_json'):
#             print(f"JSON Indexing Test Response (Status Code: {status_code_json}):")
#             print(json.dumps(response_data.get_json(), indent=2))
#         else: # Should not happen if jsonify is used
#             print(f"JSON Indexing Test Response (Status Code: {status_code_json}): {response_data}")
            
#     except RuntimeError as e:
#         if "Working outside of application context" in str(e):
#             print(f"Note: Test run finished. jsonify was called, which requires a Flask app context. "
#                   f"The core logic likely completed up to that point. Error: {e}")
#         else:
#             raise # Re-raise other RuntimeErrors


