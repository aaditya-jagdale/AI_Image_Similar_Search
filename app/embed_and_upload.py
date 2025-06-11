import json, requests, torch, time
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm  # Progress bar
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import jwt
from typing import Optional
load_dotenv()

if os.getenv("QDRANT_URL") is None or os.getenv("QDRANT_API_KEY") is None or os.getenv("SUPABASE_URL") is None or os.getenv("SUPABASE_KEY") is None:
    raise ValueError("QDRANT_URL, QDRANT_API_KEY, SUPABASE_URL, and SUPABASE_KEY must be set")

# Initialize Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# --- Load model
print("🔄 Loading Fashion CLIP model...")
processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
print("✅ Fashion CLIP model loaded.\n")

def get_embedding(url):
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    features = model.get_image_features(**inputs)
    return features[0] / features[0].norm()

def process_and_upload_data(data: list, user_id: str):
    """Process and upload data for a specific user."""
    # --- Generate vectors
    print(f"🚀 Generating image embeddings for user {user_id}...")
    points = []
    failures = []
    for i, item in enumerate(tqdm(data)):
        try:
            emb = get_embedding(item["image"])
            points.append(PointStruct(
                id=i,
                vector=emb.detach().numpy().tolist(),
                payload=item
            ))
        except Exception as e:
            failures.append(item["design_no"])
            print(f"❌ Failed on {item['design_no']}: {e}")

    print(f"\n✅ Finished embedding. Total success: {len(points)}, Failures: {len(failures)}")
    if failures:
        print("Failed designs:", failures)

    # --- Upload to Qdrant
    print("\n🔌 Connecting to Qdrant...")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    collection_name = f"textile_designs_{user_id}"
    print(f"📦 Creating collection '{collection_name}'...")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )

    print("📤 Uploading vectors to Qdrant...")
    client.upsert(collection_name=collection_name, points=points)
    print("✅ Upload complete.")
    return {
        "status": "ok", 
        "message": "Upload completed", 
        "totalItems": len(points),
        "failedItems": len(failures),
        "failedItemsList": failures,
    }
