import os
import torch
import faiss
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import numpy as np
import json
from flask import jsonify
import requests
from io import BytesIO
import fitz  # PyMuPDF
import io # Added io

# Load model & processor globally
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()

# Use CPU on Mac Mini
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # For PyTorch 1.12 or later
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device)
print(f"Using device: {device}")

# --- Configuration Constants ---
INDEX_DIR = "faiss_indices"
# METADATA_FILE = "output.json" # No longer needed
IMAGE_BASE_DIR = "" # May still be relevant if other parts use it, but not for PDF processing
BATCH_SIZE = 10
# --- Fixed filenames ---
INDEX_FILENAME = "search.faiss"
METADATA_FILENAME = "search_metadata.json"
INPUT_JSON_FILENAME = "output.json" # Added default input JSON filename

os.makedirs(INDEX_DIR, exist_ok=True)

# Define image transforms (enhanced variants)
transform_list = [
    transforms.Compose([]),  # Original image
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
    transforms.Compose([transforms.RandomRotation(25)]),
    transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2)]),
    transforms.Compose([transforms.CenterCrop(224), transforms.Resize(224)]),
    transforms.Compose([transforms.GaussianBlur(kernel_size=5)]),
    transforms.Compose([transforms.RandomPerspective(distortion_scale=0.3, p=1.0)]),
    transforms.Compose([transforms.Grayscale(num_output_channels=3)]),
]

# Updated: Accepts a PIL Image object directly
def preprocess_variants(image_obj):
    try:
        img = image_obj.convert("RGB")
        return [t(img) for t in transform_list]
    except Exception as e:
        print(f"Error processing image: {e}")
        return []

# Updated: Processes a batch of dicts containing PIL images and metadata
# Now accepts 'metadata' key instead of specific PDF keys
def process_batch(batch_image_data):
    all_variants = []
    batch_valid_metadata = []
    image_to_variants_map = []

    for item in batch_image_data:
        image_obj = item.get('image_obj') # Use .get for safety
        variants = preprocess_variants(image_obj) if image_obj else [] # Check if image_obj exists
        if variants:
            all_variants.extend(variants)
            image_to_variants_map.append(len(variants))
            # Keep metadata associated with the valid image
            batch_valid_metadata.append(item["metadata"]) # Store the original metadata dict
        # Close the PIL image object after processing its variants
        if image_obj:
             image_obj.close()


    if not all_variants:
        return [], []

    # Ensure all_variants contains PIL Images before passing to processor
    pil_images = [v for v in all_variants if isinstance(v, Image.Image)]
    if not pil_images:
        print("Warning: No valid PIL images found in batch after transformations.")
        return [], []


    inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    # Average variants per image
    embeddings = []
    start = 0
    for count in image_to_variants_map:
        # Check if start + count exceeds the bounds of features
        if start + count > features.shape[0]:
             print(f"Warning: Index out of bounds. start={start}, count={count}, features_shape={features.shape}")
             # Handle error, e.g., skip this image or adjust count
             # For now, let's skip this average calculation if bounds are wrong
             start += count
             continue

        variants_embeds = features[start:start + count]
        avg_embedding = variants_embeds.mean(dim=0)
        embeddings.append(avg_embedding.cpu().numpy())
        start += count


    return embeddings, batch_valid_metadata

# Rewritten: Accepts PDF file streams, extracts images, processes them.
def handle_build_index(pdf_files):
    if not pdf_files:
        return jsonify({"error": "No PDF files provided"}), 400

    extracted_image_data = []
    print(f"Processing {len(pdf_files)} PDF files...")

    for pdf_file in pdf_files:
        try:
            pdf_filename = pdf_file.filename
            print(f"  Extracting images from: {pdf_filename}")
            # Open PDF from stream
            pdf_doc = fitz.open(stream=pdf_file.stream.read(), filetype="pdf")

            for page_num in tqdm(range(len(pdf_doc)), desc=f"  Pages in {pdf_filename}", leave=False):
                page = pdf_doc.load_page(page_num)
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    # Load image bytes into PIL
                    try:
                        image_obj = Image.open(BytesIO(image_bytes))
                        # Store metadata relevant for PDF source
                        pdf_metadata = {
                            "source_pdf": pdf_filename,
                            "page": page_num + 1, # 1-indexed page number
                            "image_index_in_page": img_index
                        }
                        extracted_image_data.append({
                            "image_obj": image_obj,
                            "metadata": pdf_metadata
                        })
                    except Exception as pil_e:
                        print(f"    Warning: Could not load image {img_index} from page {page_num+1} of {pdf_filename}: {pil_e}")
                        continue # Skip this image

            pdf_doc.close() # Close the document after processing

        except Exception as e:
            print(f"Error processing PDF file {pdf_file.filename}: {e}")
            # Optionally decide if one error should stop the whole process
            # return jsonify({"error": f"Failed processing {pdf_file.filename}: {e}"}), 500
            continue # Continue with the next PDF

    if not extracted_image_data:
        return jsonify({"error": "No images could be extracted from the provided PDFs"}), 400

    all_embeddings = []
    valid_metadata_list = []

    print(f"Generating embeddings for {len(extracted_image_data)} extracted images...")
    # Process extracted images in batches
    for i in tqdm(range(0, len(extracted_image_data), BATCH_SIZE), desc="Embedding Batches"):
        batch_data = extracted_image_data[i:i+BATCH_SIZE]
        embeddings, valid_metadata = process_batch(batch_data)
        if embeddings:
            all_embeddings.extend(embeddings)
            valid_metadata_list.extend(valid_metadata)

    # Cleanup - Explicitly clear list to potentially help GC with large image objects
    # Although closing images in process_batch should be the main help
    del extracted_image_data

    if not all_embeddings:
        return jsonify({"error": "No embeddings generated from extracted images"}), 500

    embedding_matrix = np.vstack(all_embeddings).astype('float32')
    if embedding_matrix.size == 0:
         return jsonify({"error": "Embedding matrix is empty after processing."}), 500

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    # index_id = str(uuid.uuid4()) # Removed UUID generation
    index_filepath = os.path.join(INDEX_DIR, INDEX_FILENAME) # Use fixed name
    metadata_filepath = os.path.join(INDEX_DIR, METADATA_FILENAME) # Use fixed name

    try:
        # Ensure directory exists right before writing
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(index, index_filepath)
        # Save metadata linking index rows to source PDF/page/image index
        with open(metadata_filepath, "w") as f:
            json.dump(valid_metadata_list, f, indent=4)

        print(f"✅ FAISS index saved to '{index_filepath}'")
        print(f"✅ Metadata saved to '{metadata_filepath}'")

        return jsonify({
            "message": "Index built successfully from PDFs",
            "local_index_path": index_filepath,
            "local_metadata_path": metadata_filepath,
            "num_items_indexed": index.ntotal # Should match len(valid_metadata_list)
        }), 200
    except Exception as e:
        print(f"Error saving index or metadata: {e}")
        return jsonify({"error": f"Failed saving index/metadata: {e}"}), 500

# --- New function to build index from JSON file ---
def handle_build_index_from_json(json_data):
    """
    Builds a FAISS index from image URLs specified in the provided JSON data.

    Args:
        json_data (list): A list of objects, where each object contains an "image" URL
                          and other metadata, similar to the 'output.json' format.

    Returns:
        tuple: (Flask Response, status_code)
    """
    if not isinstance(json_data, list):
        return jsonify({"error": "JSON data must be a list of objects"}), 400

    downloaded_image_data = []
    print(f"Processing {len(json_data)} items from the provided JSON data...")

    # Wrap the loop with tqdm for progress bar
    for item in tqdm(json_data, desc="Downloading images from JSON data"):
        image_url = item.get("image")
        if not image_url or not isinstance(image_url, str):
            print(f"Warning: Skipping item due to missing or invalid 'image' URL: {item.get('design_no', 'N/A')}")
            continue

        try:
            # Download image with timeout and stream=True for potentially large files
            response = requests.get(image_url, stream=True, timeout=30) # Added timeout
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Read image content into BytesIO
            image_bytes = BytesIO(response.content)
            image_obj = Image.open(image_bytes)

            # Keep the original JSON object as metadata
            downloaded_image_data.append({
                "image_obj": image_obj,
                "metadata": item # Store the entire original item
            })

        except requests.exceptions.RequestException as req_e:
            print(f"Warning: Failed to download image from {image_url}: {req_e}")
            continue # Skip this item
        # except UnidentifiedImageError: # More specific PIL error
        #      print(f"Warning: Could not identify image format from {image_url}. Skipping.")
        #      continue # Skip this item
        except Exception as img_e:
            print(f"Warning: Error processing image from {image_url}: {img_e}")
            continue # Skip this item

    if not downloaded_image_data:
        return jsonify({"error": "No valid images could be downloaded or processed from the JSON data"}), 400

    all_embeddings = []
    valid_metadata_list = []

    print(f"Generating embeddings for {len(downloaded_image_data)} downloaded images...")
    # Process downloaded images in batches
    for i in tqdm(range(0, len(downloaded_image_data), BATCH_SIZE), desc="Embedding Batches (JSON)"):
        batch_data = downloaded_image_data[i:i+BATCH_SIZE]
        embeddings, valid_metadata = process_batch(batch_data) # Reuse existing batch processor
        if embeddings: # Ensure embeddings were actually generated
             all_embeddings.extend(embeddings)
             valid_metadata_list.extend(valid_metadata)


    # Cleanup downloaded image data
    del downloaded_image_data

    if not all_embeddings:
        return jsonify({"error": "No embeddings generated from downloaded images"}), 500

    embedding_matrix = np.vstack(all_embeddings).astype('float32')
    if embedding_matrix.size == 0:
         return jsonify({"error": "Embedding matrix is empty after processing."}), 500

    # Dimension check (should match CLIP output dimension)
    expected_dim = model.config.projection_dim
    if embedding_matrix.shape[1] != expected_dim:
        return jsonify({"error": f"Embedding dimension mismatch. Expected {expected_dim}, got {embedding_matrix.shape[1]}"}), 500


    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    index_filepath = os.path.join(INDEX_DIR, INDEX_FILENAME) # Use fixed name
    metadata_filepath = os.path.join(INDEX_DIR, METADATA_FILENAME) # Use fixed name

    try:
        # Ensure directory exists right before writing
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(index, index_filepath)
        # Save metadata (list of original JSON objects corresponding to index rows)
        with open(metadata_filepath, "w") as f:
            json.dump(valid_metadata_list, f, indent=4)

        print(f"✅ FAISS index saved to '{index_filepath}'")
        print(f"✅ Metadata saved to '{metadata_filepath}'")

        return jsonify({
            "message": f"Index built successfully from provided JSON data",
            "local_index_path": index_filepath,
            "local_metadata_path": metadata_filepath,
            "num_items_indexed": index.ntotal
        }), 200
    except Exception as e:
        print(f"Error saving index or metadata: {e}")
        return jsonify({"error": f"Failed saving index/metadata: {e}"}), 500


# Original functions below are modified or replaced above.
# ... (keep other functions if needed, or remove if entirely replaced)
