import os
import torch
import numpy as np
import clip
import faiss
from PIL import Image, UnidentifiedImageError

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Needed for FAISS on Windows

# Set device for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "pretrained_models/CIRR/RN50x4_fullft/cirr_clip_RN50x4_fullft.pt"

# Load the CLIP model
clip_model_name = 'RN50x4'
clip_model, preprocess = clip.load(clip_model_name, device=device, jit=False)
clip_state_dict = torch.load(model_path, map_location=device)
clip_model.load_state_dict(clip_state_dict["CLIP"])

# Path to the folder containing images
image_folder = "FashionIQ/resized_images/resized_images"

# Function to vectorize images
def vectorize_images(image_folder):
    image_names = []
    image_features = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path) and not image_name.startswith('._'):
            try:
                # Preprocess image
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                
                # Vectorize image
                with torch.no_grad():
                    image_feature = clip_model.encode_image(image)
                
                image_names.append(image_name)
                image_features.append(image_feature.cpu().numpy())
            except UnidentifiedImageError:
                print(f"Skipping file {image_path}: cannot identify image file")
            except Exception as e:
                print(f"Error processing file {image_path}: {e}")
        
        print(f"Processed {len(image_names)} images", end="\r")
        if len(image_names) >= 2000:  # Limit to 2000 images
            break
    return image_names, np.vstack(image_features)

# Vectorize images in the folder
image_names, image_features = vectorize_images(image_folder)

# Convert the features to float32 (required by FAISS)
image_features = image_features.astype(np.float32)

# FAISS index setup
dim = image_features.shape[1]
nlist = min(len(image_features) // 10, 20)  # Dynamically adjust clusters

print(f"Using nlist={nlist} for FAISS IVF index (dim={dim}).")
print(f"Number of training points: {image_features.shape[0]}")

# Debug: Validate the data
assert image_features.ndim == 2, f"Expected 2D array, got {image_features.ndim}D"
assert not np.any(np.isnan(image_features)), "NaN values detected in image features"
assert image_features.shape[0] >= nlist, f"Number of training points is less than nlist"

# Normalize image features (L2 normalization for cosine similarity)
faiss.normalize_L2(image_features)  # Normalize the image features

# Create FAISS IVF index for cosine similarity (inner product)
quantizer = faiss.IndexFlatIP(dim)  # Use inner product for cosine similarity
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)  # IVF with cosine similarity

# Train the index
index.train(image_features)
print("Index trained successfully.")

# Add vectors
index.add(image_features)
print(f"Added {index.ntotal} vectors to the index.")

# Save the FAISS index
try:
    faiss.write_index(index, "image_index.faiss")
    print("FAISS index saved successfully as 'image_index.faiss'.")
except Exception as e:
    print(f"Error saving FAISS index: {e}")

# Save metadata
try:
    np.savez("image_metadata.npz", names=np.array(image_names))
    print("Metadata saved successfully as 'image_metadata.npz'.")
except Exception as e:
    print(f"Error saving metadata: {e}")
