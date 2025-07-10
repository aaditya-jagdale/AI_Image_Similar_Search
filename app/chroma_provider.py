import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import asyncio
import aiohttp
from PIL import Image
import numpy as np
import io
from tqdm import tqdm
from typing import List, Dict
import logging
import json
import requests
import os
import tempfile
import shutil
from .task_tracker import task_tracker




logger = logging.getLogger(__name__)
embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")

def download_image(url: str, dest_path: str) -> str:
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # ✅ ensure directory exists
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        return dest_path
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None

def upsert_to_chroma(
    user_id: str,
    data: List[Dict],
    persist_directory: str = "./chroma",
    image_key: str = "image_url",
    id_key: str = "id",
    task_id: str = None
) -> Dict[str, int]:
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=user_id,
        embedding_function=embedding_function,
        data_loader=image_loader
    )

    temp_dir = tempfile.mkdtemp(prefix="chroma_temp_")
    upserted = 0
    failed = 0
    failures = []
    BATCH_SIZE = 32
    try:
        total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
        batch_progress = tqdm(total=total_batches, desc="Processing batches", position=0)
        completed_batches = 0
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            batch_images = []
            batch_ids = []
            batch_metadatas = []
            # Download all images in the batch first
            if task_id:
                task_tracker.update_progress(
                    task_id,
                    i,
                    len(data),
                    f"Downloading batch {i//BATCH_SIZE + 1}/{total_batches}"
                )
            download_progress = tqdm(total=len(batch), desc="Downloading images", position=1, leave=False)
            for item in batch:
                try:
                    image_path = item[image_key]
                    item_id = str(item[id_key])
                    if is_url(image_path):
                        # Download URL image to temp directory
                        temp_image_path = os.path.join(temp_dir, f"{item_id}.jpg")
                        downloaded_path = download_image(image_path, temp_image_path)
                        if downloaded_path:
                            # Load image into numpy array
                            img = Image.open(downloaded_path)
                            img_array = np.array(img)
                            batch_images.append(img_array)
                            batch_ids.append(item_id)
                            batch_metadatas.append(item)
                    else:
                        # Load local image into numpy array
                        img = Image.open(image_path)
                        img_array = np.array(img)
                        batch_images.append(img_array)
                        batch_ids.append(item_id)
                        batch_metadatas.append(item)
                except Exception as e:
                    logger.error(f"Failed to process item {item.get(id_key)}: {e}")
                    failed += 1
                    failures.append({"id": item.get(id_key), "error": str(e)})
                finally:
                    download_progress.update(1)
            download_progress.close()
            if batch_images:
                try:
                    if task_id:
                        task_tracker.update_progress(
                            task_id,
                            i + len(batch_images),
                            len(data),
                            f"Processing embeddings for batch {i//BATCH_SIZE + 1}/{total_batches}"
                        )
                    # Upsert the batch to ChromaDB
                    collection.upsert(
                        ids=batch_ids,
                        images=batch_images,
                        metadatas=batch_metadatas
                    )
                    upserted += len(batch_images)
                except Exception as e:
                    logger.error(f"Failed to upsert batch: {e}")
                    failed += len(batch_images)
                    for item_id in batch_ids:
                        failures.append({"id": item_id, "error": str(e)})
            # Clean up temp directory after each batch
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                temp_dir = tempfile.mkdtemp(prefix="chroma_temp_")
            batch_progress.update(1)
            completed_batches += 1
            # Report batch progress to task_tracker
            if task_id:
                task_tracker.update_progress(
                    task_id,
                    i + len(batch_images),
                    len(data),
                    f"Completed batch {completed_batches}/{total_batches}"
                )
        batch_progress.close()
    finally:
        # Final cleanup of temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return {
        "upserted": upserted,
        "failed": failed,
        "failures": failures,
        "total_batches": total_batches,
        "completed_batches": completed_batches if 'completed_batches' in locals() else 0
    }

def search_images(user_id: str, image_path_or_url: str, top_k: int = 10) -> List[Dict]:
    """Search for similar images in ChromaDB collection.
    
    Args:
        user_id: ID of user performing the search
        image_path_or_url: Local path or URL of query image
        top_k: Number of results to return
        
    Returns:
        List of dictionaries containing search results with metadata
    """
    # Get user's collection
    collection = chromadb.PersistentClient(path="./chroma").get_or_create_collection(
        name=user_id,
        embedding_function=embedding_function,
        data_loader=image_loader
    )

    # Load and process query image
    if image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        img = Image.open(io.BytesIO(response.content))
    else:
        img = Image.open(image_path_or_url)
    
    query_array = np.array(img)

    # Query collection
    results = collection.query(
        query_images=[query_array],
        n_results=top_k
    )

    # Format results
    formatted_results = []
    for idx, (id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
        result = {
            'id': id,
            'metadata': metadata,
            'distance': results['distances'][0][idx] if 'distances' in results else None
        }
        formatted_results.append(result)
    
    # Max allowed distance is 1
    formatted_results = [result for result in formatted_results if result['distance'] <= 1]

    return formatted_results

def search_textile_data(user_id: str, query: str, top_k: int = 10) -> List[Dict]:
    """Search for similar textile data in ChromaDB collection.
    
    Args:
        user_id: ID of user performing the search
        query: Search query
        top_k: Number of results to return

    Returns:
        List of dictionaries containing search results with metadata
    """

# if __name__ == "__main__":
#     # with open("app/textile_data.json", "r", encoding="utf-8") as f:
#     #     data: List[dict] = json.load(f)
#     # upsert_to_chroma(user_id="b8567880-c76a-4c1d-93a2-5a7d5eac62c6", data=data)
#     results = search_images(user_id="b8567880-c76a-4c1d-93a2-5a7d5eac62c6", image_path_or_url="img.webp")
#     for result in results:
#         print(f"\n\n{result}\n\n")
#         print("="*100)