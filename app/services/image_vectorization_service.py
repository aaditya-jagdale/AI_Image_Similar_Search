import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
from transformers import CLIPProcessor

class ImageVectorizerONNX:
    def __init__(self, onnx_path: str, model_id: str = "openai/clip-vit-base-patch32"):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def _load_image(self, path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")

    def encode_single(self, path: str):
        img = self._load_image(path)
        inputs = self.processor(images=img, return_tensors="np")

        pixel_values = inputs["pixel_values"]  # shape (1, 3, 224, 224)
        out = self.session.run(None, {"pixel_values": pixel_values})[0]

        # The ONNX export gives last_hidden_state, not pooled.
        # CLIP pooled = mean over sequence dim.
        pooled = out.mean(axis=1)  # shape (1, D)

        # L2 norm
        pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12)

        return pooled.astype(np.float32)

    def encode_images(self, paths):
        vectors = [self.encode_single(p) for p in paths]
        return np.vstack(vectors)
