import os
import torch
import clip
from PIL import Image
import numpy as np

# Set device for inference
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and preprocess function
clip_model, preprocess = clip.load('RN50x4', device=device, jit=False)
model_path = "pretrained_models/CIRR/RN50x4_fullft/cirr_clip_RN50x4_fullft.pt"
clip_state_dict = torch.load(model_path, map_location=device)
clip_model.load_state_dict(clip_state_dict["CLIP"])

# Directory containing the images
image_dir = "FashionIQ/resized_images/"
output_dir = "image_vectors"
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to encode an image
def encode_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Vectorize all images in the directory and save each vector to a file
for image_name in os.listdir(image_dir):
    if image_name.startswith("._"):
        continue  # Skip hidden files
    image_path = os.path.join(image_dir, image_name)
    vector = encode_image(image_path)
    if vector is not None:
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.npy")
        np.save(output_path, vector)
        print(f"Saved vector for {image_name} to {output_path}")

print("All image vectors have been saved.")