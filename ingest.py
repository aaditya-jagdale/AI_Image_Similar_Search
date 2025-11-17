import csv
import ast
import os
import requests
import tempfile
from app.services.image_vectorization_service import ImageVectorizerONNX
from app.services.simple_vector_db import SimpleVectorDB
from app.core.config import IMG_EMBEDDING_MODEL, ONNX_MODEL_PATH


CSV_PATH = "data/formatted_products_rows.csv"
MODEL_ID = IMG_EMBEDDING_MODEL # "Qdrant/clip-ViT-B-32-vision"

# --- Helper Function ---

def download_image_to_temp(image_url: str) -> str | None:
    """
    Downloads an image from a URL and saves it to a temporary local file.
    Returns the file path.
    """
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Create a temporary file
        # We use delete=False so we can close it, use it, and then delete manually
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            return tmp_file.name  # Return the path to the temp file
            
    except requests.exceptions.RequestException as e:
        print(f"  [Error] Failed to download {image_url}: {e}")
        return None

# --- Main Ingestion Logic ---

def main():
    """
    Main function to initialize services, read the CSV, and ingest data.
    """
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    print("Initializing services...")
    try:
        vectorizer = ImageVectorizerONNX(onnx_path="clip_vision_static.onnx", model_id=MODEL_ID)
    except Exception as e:
        print(f"CRITICAL: Failed to initialize ImageVectorizerONNX. Is ONNX_MODEL_PATH correct?")
        print(f"Error details: {e}")
        return
        
    db_service = SimpleVectorDB()
    print("Services initialized. Starting ingestion...")

    # Open and read the CSV file
    with open(CSV_PATH, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for i, row in enumerate(reader):
            print(f"\n--- Processing Row {i+1} ---")
            
            try:
                image_id = row['id']
                image_url = row['image']
                metadata_str = row['metadata']
                
                print(f"  ID: {image_id}")
                
                # 1. Parse metadata string
                # ast.literal_eval is safer than eval() and handles the
                # single-quote dictionary format.
                image_metadata = ast.literal_eval(metadata_str)
                
                # 2. Download image
                print(f"  Downloading: {image_url}")
                local_image_path = download_image_to_temp(image_url)
                
                if local_image_path is None:
                    continue  # Skip this row if download failed

                # 3. Generate embedding
                print(f"  Generating embedding...")
                embedding = vectorizer.encode_single(local_image_path)
                
                # 4. Add to database
                print(f"  Adding to vector database...")
                db_service.add_image(
                    embedding=embedding,
                    metadata=image_metadata,
                    id=image_id
                )
                
            except Exception as e:
                print(f"  [Error] Failed to process row {i+1} (ID: {row.get('id')}): {e}")
                
            finally:
                # 5. Clean up the temporary file
                if 'local_image_path' in locals() and local_image_path and os.path.exists(local_image_path):
                    os.remove(local_image_path)

    print("\n--- Ingestion Complete ---")

if __name__ == "__main__":
    main()