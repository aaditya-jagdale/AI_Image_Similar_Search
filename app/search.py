import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import logging
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if os.getenv("QDRANT_URL") is None or os.getenv("QDRANT_API_KEY") is None:
    logger.error("QDRANT_URL and QDRANT_API_KEY must be set in environment variables.")
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

logger.info("🔄 Loading Fashion CLIP model...")
processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
logger.info("✅ Fashion CLIP model loaded.\n")

async def get_embedding(path):
    try:
        logger.info(f"Getting embedding for path: {path}")
        if path.startswith("http"):
            image = Image.open(requests.get(path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        embedding = features[0] / features[0].norm()
        logger.info(f"Successfully got embedding for path: {path}")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for {path}: {e}")
        raise

async def search_similar(image_url, top_k=10):
    try:
        logger.info(f"Searching for similar images to {image_url} with top_k={top_k}")
        emb = await get_embedding(image_url)
        emb = emb.detach().numpy()
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        logger.info("Qdrant client initialized.")
        results = client.query_points(
                collection_name="textile_designs",
                query=emb,
                limit=top_k
            ).points
        logger.info(f"Qdrant query returned {len(results)} results.")
        if len(results) == 0:
            logger.info("No results found")
            return []
            
        logger.info("--------------------------------")
        logger.info(f"RESULTS FOR {results[0].payload.get('source_pdf', 'N/A')}: {len(results)}")
        logger.info("--------------------------------")
        return results
    except Exception as e:
        logger.error(f"Error in search_similar for {image_url}: {e}")
        raise

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(search_similar("images/JC-12.png"))
