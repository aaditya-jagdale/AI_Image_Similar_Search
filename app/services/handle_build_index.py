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
import fitz  
import io 


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()


# The device selection is modified to prioritize MPS then CPU, removing CUDA.
# if torch.backends.mps.is_available(): 
#     device = torch.device("mps")
# else:
device = torch.device("cpu")
model.to(device)
print(f"Using device: {device}")


INDEX_DIR = "faiss_indices"

IMAGE_BASE_DIR = "" 
BATCH_SIZE = 10

INDEX_FILENAME = "search.faiss"
METADATA_FILENAME = "search_metadata.json"
INPUT_JSON_FILENAME = "output.json" 

os.makedirs(INDEX_DIR, exist_ok=True)


transform_list = [
    transforms.Compose([]),  
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
    transforms.Compose([transforms.RandomRotation(25)]),
    transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2)]),
    transforms.Compose([transforms.CenterCrop(224), transforms.Resize(224)]),
    transforms.Compose([transforms.GaussianBlur(kernel_size=5)]),
    transforms.Compose([transforms.RandomPerspective(distortion_scale=0.3, p=1.0)]),
    transforms.Compose([transforms.Grayscale(num_output_channels=3)]),
]


def preprocess_variants(image_obj):
    try:
        img = image_obj.convert("RGB")
        return [t(img) for t in transform_list]
    except Exception as e:
        print(f"Error processing image: {e}")
        return []



def process_batch(batch_image_data):
    all_variants = []
    batch_valid_metadata = []
    image_to_variants_map = []

    for item in batch_image_data:
        image_obj = item.get('image_obj') 
        variants = preprocess_variants(image_obj) if image_obj else [] 
        if variants:
            all_variants.extend(variants)
            image_to_variants_map.append(len(variants))
            
            batch_valid_metadata.append(item["metadata"]) 
        
        if image_obj:
             image_obj.close()


    if not all_variants:
        return [], []

    
    pil_images = [v for v in all_variants if isinstance(v, Image.Image)]
    if not pil_images:
        print("Warning: No valid PIL images found in batch after transformations.")
        return [], []


    inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    
    embeddings = []
    start = 0
    for count in image_to_variants_map:
        
        if start + count > features.shape[0]:
             print(f"Warning: Index out of bounds. start={start}, count={count}, features_shape={features.shape}")
             
             
             start += count
             continue

        variants_embeds = features[start:start + count]
        avg_embedding = variants_embeds.mean(dim=0)
        embeddings.append(avg_embedding.cpu().numpy())
        start += count


    return embeddings, batch_valid_metadata


def handle_build_index(pdf_files):
    if not pdf_files:
        return jsonify({"error": "No PDF files provided"}), 400

    extracted_image_data = []
    print(f"Processing {len(pdf_files)} PDF files...")

    for pdf_file in pdf_files:
        try:
            pdf_filename = pdf_file.filename
            print(f"  Extracting images from: {pdf_filename}")
            
            pdf_doc = fitz.open(stream=pdf_file.stream.read(), filetype="pdf")

            for page_num in tqdm(range(len(pdf_doc)), desc=f"  Pages in {pdf_filename}", leave=False):
                page = pdf_doc.load_page(page_num)
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    try:
                        image_obj = Image.open(BytesIO(image_bytes))
                        
                        pdf_metadata = {
                            "source_pdf": pdf_filename,
                            "page": page_num + 1, 
                            "image_index_in_page": img_index
                        }
                        extracted_image_data.append({
                            "image_obj": image_obj,
                            "metadata": pdf_metadata
                        })
                    except Exception as pil_e:
                        print(f"    Warning: Could not load image {img_index} from page {page_num+1} of {pdf_filename}: {pil_e}")
                        continue 

            pdf_doc.close() 

        except Exception as e:
            print(f"Error processing PDF file {pdf_file.filename}: {e}")
            
            
            continue 

    if not extracted_image_data:
        return jsonify({"error": "No images could be extracted from the provided PDFs"}), 400

    all_embeddings = []
    valid_metadata_list = []

    print(f"Generating embeddings for {len(extracted_image_data)} extracted images...")
    
    for i in tqdm(range(0, len(extracted_image_data), BATCH_SIZE), desc="Embedding Batches"):
        batch_data = extracted_image_data[i:i+BATCH_SIZE]
        embeddings, valid_metadata = process_batch(batch_data)
        if embeddings:
            all_embeddings.extend(embeddings)
            valid_metadata_list.extend(valid_metadata)

    
    
    del extracted_image_data

    if not all_embeddings:
        return jsonify({"error": "No embeddings generated from extracted images"}), 500

    embedding_matrix = np.vstack(all_embeddings).astype('float32')
    if embedding_matrix.size == 0:
         return jsonify({"error": "Embedding matrix is empty after processing."}), 500

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    
    index_filepath = os.path.join(INDEX_DIR, INDEX_FILENAME) 
    metadata_filepath = os.path.join(INDEX_DIR, METADATA_FILENAME) 

    try:
        
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(index, index_filepath)
        
        with open(metadata_filepath, "w") as f:
            json.dump(valid_metadata_list, f, indent=4)

        print(f"✅ FAISS index saved to '{index_filepath}'")
        print(f"✅ Metadata saved to '{metadata_filepath}'")

        return jsonify({
            "message": "Index built successfully from PDFs",
            "local_index_path": index_filepath,
            "local_metadata_path": metadata_filepath,
            "num_items_indexed": index.ntotal 
        }), 200
    except Exception as e:
        print(f"Error saving index or metadata: {e}")
        return jsonify({"error": f"Failed saving index/metadata: {e}"}), 500


def handle_build_index_from_json(json_data):
    """
    Builds a FAISS index from image URLs specified in the provided JSON data.
    Processes images in batches to manage memory usage.

    Args:
        json_data (list): A list of objects, where each object contains an "image" URL
                          and other metadata, similar to the 'output.json' format.

    Returns:
        tuple: (Flask Response, status_code)
    """
    if not isinstance(json_data, list):
        return jsonify({"error": "JSON data must be a list of objects"}), 400

    all_embeddings = []
    valid_metadata_list = []
    total_items = len(json_data)

    print(f"Processing {total_items} items from the provided JSON data in batches...")

    
    for i in tqdm(range(0, total_items, BATCH_SIZE), desc="Processing Item Batches"):
        current_batch_json_items = json_data[i:i + BATCH_SIZE]
        current_batch_image_data = [] 

        
        for item in current_batch_json_items:
            image_url = item.get("image")
            if not image_url or not isinstance(image_url, str):
                print(f"Warning: Skipping item due to missing or invalid 'image' URL in current batch: {item.get('design_no', 'N/A')}")
                continue

            try:
                response = requests.get(image_url, stream=True, timeout=30)
                response.raise_for_status()
                image_bytes = BytesIO(response.content)
                image_obj = Image.open(image_bytes)
                current_batch_image_data.append({
                    "image_obj": image_obj,
                    "metadata": item
                })
            except requests.exceptions.RequestException as req_e:
                print(f"Warning: Failed to download image from {image_url}: {req_e}")
                continue
            except Exception as img_e: 
                print(f"Warning: Error processing image from {image_url}: {img_e}")
                continue

        if not current_batch_image_data:
            continue 

        embeddings, valid_metadata = process_batch(current_batch_image_data)

        if embeddings: 
            all_embeddings.extend(embeddings)
            valid_metadata_list.extend(valid_metadata)
        
    if not all_embeddings:
        return jsonify({"error": "No embeddings could be generated from the provided JSON data. Check image URLs and formats."}), 400

    if not all_embeddings: 
        return jsonify({"error": "No embeddings generated from downloaded images"}), 500

    embedding_matrix = np.vstack(all_embeddings).astype('float32')
    if embedding_matrix.size == 0:
         return jsonify({"error": "Embedding matrix is empty after processing."}), 500

    
    expected_dim = model.config.projection_dim
    if embedding_matrix.shape[1] != expected_dim:
        return jsonify({"error": f"Embedding dimension mismatch. Expected {expected_dim}, got {embedding_matrix.shape[1]}"}), 500


    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    index_filepath = os.path.join(INDEX_DIR, INDEX_FILENAME) 
    metadata_filepath = os.path.join(INDEX_DIR, METADATA_FILENAME) 

    try:
        
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(index, index_filepath)
        
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




