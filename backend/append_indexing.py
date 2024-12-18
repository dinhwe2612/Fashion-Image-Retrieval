import os
import torch
import numpy as np
import clip
import faiss
from PIL import Image, UnidentifiedImageError

# Set device for inference
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set model path
model_path = "pretrained_models/CIRR/RN50x4_fullft/cirr_clip_RN50x4_fullft.pt"
fine_tuned_model_path = "fine_tuned_models/tuned_clip_best.pt"

# Load the CLIP model
clip_model_name = 'RN50x4'
clip_model, preprocess = clip.load(clip_model_name, device=device, jit=False)

clip_state_dict = torch.load(model_path, map_location=device)
# clip_state_dict = torch.load(fine_tuned_model_path, map_location=device)
clip_model.load_state_dict(clip_state_dict["CLIP"])

# Paths
image_folder = "FashionIQ/resized_images/resized_images"
index_path = "image_index.faiss"
metadata_path = "image_metadata.npz"

# Load existing FAISS index and metadata
def load_existing_index(index_path, metadata_path):
    try:
        index = faiss.read_index(index_path)
        metadata = np.load(metadata_path, allow_pickle=True)
        image_names = metadata['names'].tolist()
        print("Loaded existing FAISS index and metadata.")
        return index, image_names
    except Exception as e:
        print(f"Error loading existing index or metadata: {e}")
        return None, []

# Vectorize new images
def vectorize_new_images(image_folder, existing_names):
    new_image_names = []
    new_image_features = []

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path) and image_name not in existing_names and not image_name.startswith('._'):
            try:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_feature = clip_model.encode_image(image)
                new_image_names.append(image_name)
                new_image_features.append(image_feature.cpu().numpy())
            except UnidentifiedImageError:
                print(f"Skipping file {image_path}: cannot identify image file")
            except Exception as e:
                print(f"Error processing file {image_path}: {e}")

    if new_image_features:
        new_image_features = np.vstack(new_image_features).astype(np.float32)
        faiss.normalize_L2(new_image_features)  # Normalize for cosine similarity
    
    print(f"Vectorized {len(new_image_names)} new images.")
    return new_image_names, new_image_features

# Append new features to FAISS index and metadata
def append_to_index(index, image_features):
    index.add(image_features)
    print(f"Added {image_features.shape[0]} new vectors to the index.")
    return index

# Save updated index and metadata
def save_updated_index(index, index_path, image_names, metadata_path):
    try:
        faiss.write_index(index, index_path)
        print(f"FAISS index updated and saved to '{index_path}'.")
        np.savez(metadata_path, names=np.array(image_names))
        print(f"Metadata updated and saved to '{metadata_path}'.")
    except Exception as e:
        print(f"Error saving updated index or metadata: {e}")

# Main logic
def main():
    # Load existing index and metadata
    index, existing_image_names = load_existing_index(index_path, metadata_path)
    if index is None:
        print("No existing index found. Exiting.")
        return

    # Vectorize new images
    new_image_names, new_image_features = vectorize_new_images(image_folder, existing_image_names)
    
    if new_image_features.size > 0:
        # Append new features to the FAISS index
        index = append_to_index(index, new_image_features)
        
        # Update image names metadata
        updated_image_names = existing_image_names + new_image_names
        
        # Save updated index and metadata
        save_updated_index(index, index_path, updated_image_names, metadata_path)
    else:
        print("No new images to process.")

if __name__ == "__main__":
    main()
