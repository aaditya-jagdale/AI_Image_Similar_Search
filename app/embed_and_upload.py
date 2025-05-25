import json, requests, torch, time
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm  # Progress bar
from dotenv import load_dotenv
import os
load_dotenv()

if os.getenv("QDRANT_URL") is None or os.getenv("QDRANT_API_KEY") is None:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

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

# --- Load textile JSON
print("📄 Loading textile data...")
with open("sample_data.json") as f:
    data = json.load(f)
print(f"✅ Loaded {len(data)} textile entries.\n")

# --- Generate vectors
print("🚀 Generating image embeddings...")
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
        failures.append(item["design no"])
        print(f"❌ Failed on {item['design no']}: {e}")

print(f"\n✅ Finished embedding. Total success: {len(points)}, Failures: {len(failures)}")
if failures:
    print("Failed designs:", failures)

# --- Upload to Qdrant
print("\n🔌 Connecting to Qdrant...")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

print("📦 Recreating collection 'textile_designs'...")
client.recreate_collection(
    collection_name="textile_designs",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
)

print("📤 Uploading vectors to Qdrant...")
client.upsert(collection_name="textile_designs", points=points)
print("✅ Upload complete.")
