import json
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import models
import chromadb

ALL_ITEMS_JSON="data/output/pdf_extracted_data.json"
LIMIT = 99999999

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("textile_images")

preprocess = transforms.Compose([
    transforms.Resize(256),       # 1. Shrink the 500x500 crop a bit
    transforms.CenterCrop(224),   # 2. Grab the center 224x224 (Model's favorite size)
    transforms.ToTensor(),        # 3. Turn pixels into a math grid (Tensor)
    transforms.Normalize(         # 4. Standardize colors 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 1. Load the brain with pre-learned knowledge (weights)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 2. "Decapitate" it! Replace the final 'fc' (fully connected) layer
model.fc = nn.Identity()

# 3. Set to 'evaluation mode' (tells the brain we are using it, not teaching it)
model.eval()

# Read JSON file
with open(ALL_ITEMS_JSON, "r") as f:
    item_list = json.load(f)
    for item in item_list[:LIMIT]:
        #access the item
        item_image = f"data/output/{item['image_path']}"
        # Crop top left corner of the image into 500x500 square
        crop_size = 500
        item_image = Image.open(item_image)
        item_image = item_image.crop((item_image.width - crop_size, item_image.height - crop_size, item_image.width, item_image.height))
        img_tensor = preprocess(item_image)
        img_batch = img_tensor.unsqueeze(0)
        with torch.no_grad():
            embeddings = model(img_batch)
        
        vector = embeddings.squeeze(0).tolist()
        collection.add(
            ids=[item['design_no']],
            embeddings=[vector],
            metadatas=[{
                'source_pdf': item['source_pdf'],
                'design_no': item['design_no'],
                'width': item['width'],
                'stock': item['stock'],
                'gsm': item['gsm'],
                'image_path': item['image_path'],
            }]
        )
        print(f"[INFO] Added {item['design_no']} to collection successfully")