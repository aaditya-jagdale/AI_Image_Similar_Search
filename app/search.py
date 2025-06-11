import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import logging
import jwt
from typing import Optional, List
from supabase import create_client, Client
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if os.getenv("QDRANT_URL") is None or os.getenv("QDRANT_API_KEY") is None or os.getenv("SUPABASE_URL") is None or os.getenv("SUPABASE_KEY") is None:
    raise ValueError("QDRANT_URL, QDRANT_API_KEY, SUPABASE_URL, and SUPABASE_KEY must be set")

# Initialize Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def validate_token(token: str) -> Optional[str]:
    """Validate the Supabase JWT token and return the user ID if valid."""
    try:
        # Decode the JWT token without verification (Supabase tokens are already verified)
        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded.get("sub")  # 'sub' is the user ID in Supabase JWT
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        return None

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

async def search_similar(image_url: str, client: QdrantClient, user_id: str, top_k: int = 10) -> List:
    """
    Search for similar images in a user-specific collection.
    
    Args:
        image_url: URL or path to the image to search for
        client: Qdrant client instance
        user_id: User ID to identify the collection
        top_k: Number of results to return
        
    Returns:
        List of similar images with their metadata
    """
    try:
        logger.info(f"Searching for similar images to {image_url} with top_k={top_k}")
        emb = await get_embedding(image_url)
        emb = emb.detach().numpy()
        
        collection_name = f"textile_designs_{user_id}"
        logger.info(f"Querying collection '{collection_name}'")
        
        results = client.query_points(
            collection_name=collection_name,
            query=emb,
            limit=top_k
        ).points
        
        logger.info(f"Qdrant query returned {len(results)} results.")
        if len(results) == 0:
            logger.info("No results found")
            return []
            
        logger.info("--------------------------------")
        logger.info(f"RESULTS FOR USER {user_id}: {len(results)}")
        logger.info("--------------------------------")
        
        for result in results:
            design_no = result.payload.get('design no', 'N/A')
            width = result.payload.get('width', 'N/A')
            stock = result.payload.get('stock', 'N/A')
            GSM = result.payload.get('GSM', 'N/A')
            image = result.payload.get('image', 'N/A')
            logger.info(f"Design No: {design_no}\nWidth: {width}\nStock: {stock}\nGSM: {GSM}\nImage: {image}")
            logger.info("--------------------------------")

        return results
    except Exception as e:
        logger.error(f"Error in search_similar for {image_url}: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    token = input("Enter your Supabase session token: ")
    user_id = validate_token(token)
    
    if not user_id:
        logger.error("❌ Invalid token")
        exit(1)
        
    logger.info(f"✅ Valid token for user {user_id}")
    
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    import asyncio
    asyncio.run(search_similar("images/JCY-9.png", qdrant_client, user_id))
