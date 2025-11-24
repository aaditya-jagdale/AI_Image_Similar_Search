from upstash_vector import Index
import os
import numpy as np
import torch
from torchvision import transforms
from transformers import CLIPModel
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

LIMIT=10

index = Index(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
IMAGE_DIR="data/output/images"
images = []
vectors = []

#Load Images from dir
for filename in os.listdir(IMAGE_DIR):
    if len(images) < LIMIT:
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(IMAGE_DIR, filename)
            img = Image.open(file_path) 

            if img is not None:
                images.append(img)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform_image(image):
    image = image.convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = model.get_image_features(pixel_values=image)
    embedding = features.squeeze().cpu().numpy()
    return embedding.astype(np.float32)

# -----------------------------------------------

# Extract embedding for each image and insert into index
for i, entry in enumerate(tqdm(images)):
    embedding = transform_image(entry)

    # Store the vectors in an array to upsert all of them at the same time.
    vectors.append({"vector": embedding.tolist(), "id": i})

for i in tqdm(range(0, len(vectors), LIMIT)):
    index.upsert(vectors[i:i+LIMIT])