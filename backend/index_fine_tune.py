import os
import torch
import numpy as np
import clip
import faiss
from PIL import Image, UnidentifiedImageError

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix FAISS on Windows

device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to fine-tuned model (your model)
fine_tuned_model_path = "fine_tuned_models/tuned_clip_best.pt"

# Load the base CLIP model and your fine-tuned model
clip_model_name = 'RN50x4'
base_clip_model, preprocess = clip.load(clip_model_name, device=device, jit=False)

# Load fine-tuned weights
fine_tuned_state_dict = torch.load(fine_tuned_model_path, map_location=device)

try:
    # Replace the base CLIP model weights with fine-tuned weights
    base_clip_model.load_state_dict(fine_tuned_state_dict["CLIP"])
    print("Fine-tuned model loaded successfully.")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    exit()

# Path to images
image_folder = "FashionIQ/resized_images/resized_images"

def vectorize_images(image_folder):
    image_names = []
    image_features = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path) and not image_name.startswith('._'):
            try:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_feature = base_clip_model.encode_image(image)
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

# Vectorize images
image_names, image_features = vectorize_images(image_folder)
image_features = image_features.astype(np.float32)

# Setup FAISS index for cosine similarity
dim = image_features.shape[1]
nlist = min(len(image_features) // 10, 20)

print(f"Using nlist={nlist} for FAISS IVF index (dim={dim}).")
faiss.normalize_L2(image_features)
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

index.train(image_features)
print("Index trained successfully.")
index.add(image_features)
print(f"Added {index.ntotal} vectors to the index.")

# Save FAISS index and metadata
faiss.write_index(index, "fine_tuned_image_index.faiss")
print("FAISS index saved as 'fine_tuned_image_index.faiss'.")
np.savez("fine_tuned_image_metadata.npz", names=np.array(image_names))
print("Metadata saved as 'fine_tuned_image_metadata.npz'.")
