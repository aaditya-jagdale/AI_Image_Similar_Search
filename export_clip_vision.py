# export_clip_vision_fixed.py
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

model_id = "openai/clip-vit-base-patch32"
# load vision submodel (only the visual encoder)
vision_model = CLIPModel.from_pretrained(model_id).vision_model.eval()

processor = CLIPProcessor.from_pretrained(model_id)

# create dummy input of the exact size CLIP expects (224x224)
img = Image.new("RGB", (224, 224), color=(128, 128, 128))
inputs = processor(images=img, return_tensors="pt")
pixel_values = inputs["pixel_values"]  # shape (1,3,224,224)

# Export with static H/W (no dynamic axes)
torch.onnx.export(
    vision_model,
    (pixel_values,),
    "clip_vision_static.onnx",
    opset_version=17,
    input_names=["pixel_values"],
    output_names=["last_hidden_state"],
    dynamic_axes={"pixel_values": {0: "batch"}},  # only batch is dynamic
    do_constant_folding=True,
)
print("Exported clip_vision_static.onnx")
