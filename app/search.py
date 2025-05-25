import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
load_dotenv()

if os.getenv("QDRANT_URL") is None or os.getenv("QDRANT_API_KEY") is None:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

print("🔄 Loading Fashion CLIP model...")
processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
print("✅ Fashion CLIP model loaded.\n")

def get_embedding(path):
    if path.startswith("http"):
        image = Image.open(requests.get(path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    features = model.get_image_features(**inputs)
    return features[0] / features[0].norm()

def search_similar(image_url, top_k=10):
    emb = get_embedding(image_url).detach().numpy()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    results = client.query_points(
            collection_name="textile_designs",
            query=emb,
            limit=top_k
        ).points
    if len(results) == 0:
        print("No results found")
        return
        
    print("--------------------------------")
    print(f"RESULTS FOR {results[0].payload['source_pdf']}: ", len(results))
    print("--------------------------------")
    return results

# if __name__ == "__main__":
#     search_similar("images/JC-12.png")
