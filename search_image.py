from PIL import Image
import chromadb
from torchvision import transforms
import torch
from torchvision import models
import torch.nn as nn


client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("textile_images")
# 1. Load the brain with pre-learned knowledge (weights)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 2. "Decapitate" it! Replace the final 'fc' (fully connected) layer
model.fc = nn.Identity()

# 3. Set to 'evaluation mode' (tells the brain we are using it, not teaching it)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),     
    transforms.CenterCrop(224), 
    transforms.ToTensor(),      
    transforms.Normalize(       
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# send image file instead
async def get_image_embedding(image_file: Image.Image):
    # Crop top left corner of the image into 500x500 square
    crop_size = 500
    item_image = image_file.convert('RGB')
    item_image = item_image.crop((item_image.width - crop_size, item_image.height - crop_size, item_image.width, item_image.height))
    img_tensor = preprocess(item_image)
    img_batch = img_tensor.unsqueeze(0)
    with torch.no_grad():
        embeddings = model(img_batch)
    
    vector = embeddings.squeeze(0).tolist()
    results = collection.query(query_embeddings=vector, n_results=5)
    return results
