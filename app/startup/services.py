"""
Service initialization module.
All heavy services (models, databases, etc.) are initialized here at server startup.
"""
import os
import numpy as np
import torch
from typing import Optional
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel
from upstash_vector import Index
import logging

logger = logging.getLogger(__name__)

# Global service instances
upstash_index: Optional[Index] = None
clip_model: Optional[CLIPModel] = None
clip_preprocess: Optional[transforms.Compose] = None


def initialize_upstash_index() -> Optional[Index]:
    """Initialize Upstash Vector Index."""
    global upstash_index
    
    upstash_url = os.getenv("UPSTASH_VECTOR_REST_URL")
    upstash_token = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    
    if not upstash_url or not upstash_token:
        logger.warning("UPSTASH_VECTOR_REST_URL or UPSTASH_VECTOR_REST_TOKEN not set")
        upstash_index = None
        return None
    
    try:
        upstash_index = Index(url=upstash_url, token=upstash_token)
        logger.info("Upstash Index initialized successfully")
        return upstash_index
    except Exception as e:
        logger.error(f"Failed to initialize Upstash Index: {e}")
        upstash_index = None
        return None


def initialize_clip_model() -> tuple[Optional[CLIPModel], Optional[transforms.Compose]]:
    """Initialize CLIP model and preprocessing pipeline."""
    global clip_model, clip_preprocess
    
    try:
        logger.info("Loading CLIP model (this may take a moment)...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()  # Set to evaluation mode
        
        # Preprocessing pipeline (same as notebook)
        # Note: ToTensor() handles PIL Image conversion, but we ensure RGB conversion happens first
        clip_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        logger.info("CLIP model initialized successfully")
        return clip_model, clip_preprocess
    except Exception as e:
        logger.error(f"Failed to initialize CLIP model: {e}")
        clip_model = None
        clip_preprocess = None
        return None, None


def get_image_embedding(image: Image.Image) -> np.ndarray:
    """
    Generate embedding for an image using the pre-loaded CLIP model.
    
    Args:
        image: PIL Image object (can be in any mode - will be converted to RGB)
        
    Returns:
        numpy array of the embedding
        
    Raises:
        RuntimeError: If CLIP model is not initialized
        ValueError: If image cannot be converted to RGB
    """
    if clip_model is None or clip_preprocess is None:
        raise RuntimeError("CLIP model not initialized. Please check server startup logs.")
    
    # Convert to RGB - this MUST happen BEFORE ToTensor() to avoid channel mismatch
    # PIL's convert("RGB") handles RGBA, P (palette), L (grayscale), and other modes correctly
    original_mode = image.mode
    if image.mode != "RGB":
        try:
            image = image.convert("RGB")
        except Exception as e:
            raise ValueError(
                f"Failed to convert image from mode '{original_mode}' to RGB: {e}"
            )
    
    # Verify the image is now RGB
    if image.mode != "RGB":
        raise ValueError(
            f"Image conversion failed. Expected RGB mode, got {image.mode}"
        )
    
    # Preprocess the image (now guaranteed to be RGB with 3 channels)
    image_tensor = clip_preprocess(image)
    
    # Verify tensor has correct shape (should be [3, 224, 224])
    if image_tensor.shape[0] != 3:
        raise ValueError(
            f"Expected 3-channel RGB tensor, but got {image_tensor.shape[0]} channels. "
            f"Original image mode was: {original_mode}, converted mode: {image.mode}"
        )
    
    image_batch = image_tensor.unsqueeze(0)
    
    # Generate embedding
    with torch.no_grad():
        features = clip_model.get_image_features(pixel_values=image_batch)
    
    embedding = features.squeeze().cpu().numpy()
    return embedding.astype(np.float32)


def initialize_all_services():
    """Initialize all services at server startup."""
    global upstash_index, clip_model, clip_preprocess
    
    logger.info("Initializing services...")
    
    # Initialize Upstash Index
    upstash_index = initialize_upstash_index()
    
    # Initialize CLIP model
    clip_model, clip_preprocess = initialize_clip_model()
    
    # Verify critical services
    if upstash_index is None:
        logger.warning("Upstash Index is not available. Image search will not work.")
    
    if clip_model is None:
        logger.warning("CLIP model is not available. Image search will not work.")
    
    if upstash_index and clip_model:
        logger.info("All services initialized successfully")
    else:
        logger.error("Some services failed to initialize. Check logs above.")


def cleanup_services():
    """Cleanup services on server shutdown."""
    global upstash_index, clip_model, clip_preprocess
    
    logger.info("Cleaning up services...")
    
    # Clear model from memory
    if clip_model is not None:
        del clip_model
        clip_model = None
    
    if clip_preprocess is not None:
        clip_preprocess = None
    
    # Upstash Index doesn't need explicit cleanup
    upstash_index = None
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Services cleaned up")


def get_upstash_index() -> Optional[Index]:
    """Get the initialized Upstash Index instance."""
    return upstash_index


def is_services_ready() -> bool:
    """Check if all required services are initialized."""
    return upstash_index is not None and clip_model is not None

