import os
import io
import torch
import faiss
import requests
import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel, PreTrainedModel
from torchvision import transforms
from tqdm import tqdm
from typing import Tuple, List, Optional
from flask import current_app

class ClipService:
    """Handles loading and using the CLIP model."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[PreTrainedModel] = None

    def load(self):
        """Loads the CLIP model and processor."""
        try:
            print(f"Loading CLIP model and processor ({self.model_name})...")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            print("CLIP model and processor loaded successfully.")
        except Exception as e:
            print(f"Error loading CLIP model/processor: {e}")
            self.model = None
            self.processor = None
            # Optionally re-raise or handle more gracefully
            raise RuntimeError(f"Failed to load CLIP model: {e}")

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generates a normalized embedding for a single image."""
        if not self.model or not self.processor:
            raise RuntimeError("CLIP model or processor not loaded.")
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
                # Normalize the embedding
                embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().astype('float32')
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")


class FaissIndexService:
    """Handles loading, building, and searching the FAISS index."""
    def __init__(self, index_path: str, ids_path: str, clip_service: ClipService):
        self.index_path = index_path
        self.ids_path = ids_path
        self.clip_service = clip_service
        self.index: Optional[faiss.Index] = None
        self.product_ids: List[str] = []

    def load_index(self) -> bool:
        """Loads the FAISS index and product IDs if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.ids_path):
            try:
                print(f"Loading FAISS index from {self.index_path}...")
                self.index = faiss.read_index(self.index_path)
                print(f"Loading product IDs from {self.ids_path}...")
                with open(self.ids_path, "r") as f:
                    self.product_ids = [line.strip() for line in f]
                print(f"Index ({self.index.ntotal} items) and product IDs ({len(self.product_ids)} items) loaded.")
                if self.index.ntotal != len(self.product_ids):
                     print("Warning: Mismatch between index size and number of product IDs.")
                     # Decide how to handle mismatch - here we clear and force rebuild later
                     # self.index = None
                     # self.product_ids = []
                     # return False
                return True
            except Exception as e:
                print(f"Error loading index or IDs: {e}")
                self.index = None
                self.product_ids = []
                return False
        else:
            print("Index file or IDs file not found.")
            self.index = None
            self.product_ids = []
            return False

    def build_index(self, image_paths: List[str]) -> Tuple[int, str]:
        """Builds or rebuilds the FAISS index from a list of image file paths."""
        if not self.clip_service.model or not self.clip_service.processor:
            raise RuntimeError("CLIP model or processor not loaded for index building.")

        if not image_paths:
            raise ValueError("No valid image paths provided for indexing.")

        print(f"Processing {len(image_paths)} provided images. Building index...")

        all_embeddings = []
        current_product_ids = []
        skipped_count = 0

        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                image = Image.open(img_path).convert("RGB")
                product_embedding = self.clip_service.get_image_embedding(image) # Get single embedding per image
                all_embeddings.append(product_embedding.squeeze()) # Squeeze to remove batch dim
                current_product_ids.append(os.path.basename(img_path))
            except FileNotFoundError:
                 print(f"Warning: Skipping non-existent file {img_path}")
                 skipped_count += 1
                 continue # Skip to next image
            except UnidentifiedImageError:
                print(f"Warning: Skipping non-image file {img_path}")
                skipped_count += 1
            except Exception as e:
                print(f"Warning: Skipping image {img_path} due to error: {e}")
                skipped_count += 1
                continue # Skip to next image on embedding error

        if not all_embeddings:
             raise ValueError("No embeddings could be generated from the provided images.")

        embedding_matrix = np.vstack(all_embeddings) # vstack expects NxD arrays, squeeze ensures this

        # Ensure the matrix is float32 for FAISS
        if embedding_matrix.dtype != np.float32:
             embedding_matrix = embedding_matrix.astype('float32')

        # Create FAISS index (using IndexFlatL2, simple L2 distance)
        dimension = embedding_matrix.shape[1]
        new_index = faiss.IndexFlatL2(dimension)
        new_index.add(embedding_matrix)

        # Save index and product IDs atomically (optional improvement)
        temp_index_path = self.index_path + ".tmp"
        temp_ids_path = self.ids_path + ".tmp"

        try:
            faiss.write_index(new_index, temp_index_path)
            with open(temp_ids_path, "w") as f:
                for pid in current_product_ids:
                    f.write(pid + "\n")

            # Rename temp files to final names
            os.replace(temp_index_path, self.index_path)
            os.replace(temp_ids_path, self.ids_path)

        except Exception as e:
            # Clean up temp files on error
            if os.path.exists(temp_index_path): os.remove(temp_index_path)
            if os.path.exists(temp_ids_path): os.remove(temp_ids_path)
            raise IOError(f"Failed to save index/IDs: {e}")
        finally: # Add colon here
              # Ensure temp files are removed even if os.replace fails somehow
              if os.path.exists(temp_index_path): os.remove(temp_index_path)
              if os.path.exists(temp_ids_path): os.remove(temp_ids_path)


        # Update service state
        self.index = new_index
        self.product_ids = current_product_ids
        num_items = len(self.product_ids)

        print(f"✅ FAISS index built with {num_items} items and saved.")
        status_message = f"Index built successfully with {num_items} items."
        if skipped_count > 0:
            status_message += f" Skipped {skipped_count} files."
        return num_items, status_message


    def search_similar(self, query_embedding: np.ndarray, k: int) -> List[dict]:
        """Searches the index for the top k similar items."""
        if self.index is None or not self.product_ids:
            raise RuntimeError("Index not built or loaded.")
        if self.index.ntotal == 0:
             return [] # Return empty if index is empty

        actual_k = min(k, self.index.ntotal) # Ensure k is not larger than the index size
        if actual_k == 0:
             return []

        try:
            distances, indices = self.index.search(query_embedding, k=actual_k)
            results = []
            for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if 0 <= idx < len(self.product_ids): # Check index bounds
                    results.append({
                        "rank": rank + 1,
                        "id": self.product_ids[idx],
                        "distance": float(dist) # Convert numpy float32 to python float
                    })
                else:
                    # This shouldn't happen with a correctly built index but good to log
                    print(f"Warning: Search returned invalid index {idx} (out of bounds for {len(self.product_ids)})")
            return results
        except Exception as e:
            print(f"FAISS search error: {e}")
            raise RuntimeError(f"Image search failed internally (FAISS error): {e}")

def fetch_image_from_url(image_url: str, timeout: int = 15) -> Image.Image:
    """Fetches and opens an image from a URL."""
    try:
        print(f"Fetching image from URL: {image_url}")
        response = requests.get(image_url, stream=True, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get('content-type')
        if content_type and not content_type.lower().startswith('image/'):
            raise ValueError(f"URL content type is '{content_type}', not an image.")

        image_data = io.BytesIO(response.content)
        query_image = Image.open(image_data).convert("RGB")
        return query_image
    except requests.exceptions.Timeout:
        print(f"Error: Timeout while fetching image from URL {image_url}")
        raise TimeoutError(f"Timeout while fetching image from URL: {image_url}")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else None
        print(f"Error downloading image from URL {image_url}: {e}")
        if status_code:
             raise ConnectionError(f"Failed to download image (HTTP {status_code}): {image_url}")
        else:
             raise ConnectionError(f"Failed to download image: {e}")
    except UnidentifiedImageError:
        print(f"Error: Could not identify image data from URL {image_url}")
        raise ValueError("Could not decode image data from URL. Is it a valid image file?")
    except Exception as e:
         print(f"Unexpected error fetching image: {e}")
         raise RuntimeError(f"Unexpected error fetching image: {e}")

# --- Helpers to access services from application context ---

def get_clip_service() -> 'ClipService': # Forward reference using string
    """Gets the ClipService instance from the Flask app context."""
    return current_app.extensions['clip_service']

def get_faiss_service() -> 'FaissIndexService': # Forward reference
    """Gets the FaissIndexService instance from the Flask app context."""
    return current_app.extensions['faiss_service'] 