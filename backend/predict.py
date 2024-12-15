import os
import torch
import numpy as np
import clip
import faiss
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from io import BytesIO
from combiner import Combiner
from data_utils import targetpad_transform
import base64
from typing import Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Needed for FAISS on Windows

# Set device for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "pretrained_models/CIRR/RN50x4_fullft/cirr_clip_RN50x4_fullft.pt"
combiner_path = "pretrained_models/CIRR/RN50x4_fullft/cirr_comb_RN50x4_fullft.pt"

# Fine-tuned model index and metadata paths
fine_tuned_index_path = "fine_tuned_image_index.faiss"
fine_tuned_metadata_path = "fine_tuned_image_metadata.npz"

fine_tuned_model_path = "fine_tuned_models/tuned_clip_best.pt"

# Initialize FastAPI app
app = FastAPI()

# Load Fine-Tuned CLIP model
def load_fine_tuned_clip_model():
    clip_model_name = 'RN50x4'
    clip_model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_state_dict = torch.load(fine_tuned_model_path, map_location=device)
    clip_model.load_state_dict(clip_state_dict["CLIP"])  # Load tuned_clip_best.pt weights
    clip_model.eval()
    return clip_model, preprocess

# Load Fine-Tuned CLIP model
# def load_fine_tuned_clip_model():
#     clip_model_name = 'RN50x4'
#     clip_model, preprocess = clip.load(clip_model_name, device=device, jit=False)

#     # Load fine-tuned weights
#     clip_state_dict = torch.load(fine_tuned_model_path, map_location=device)

#     # Check for the "CLIP" key and extract it
#     if "CLIP" in clip_state_dict:
#         clip_state_dict = clip_state_dict["CLIP"]  # Extract the actual model weights
    
#     # Load state dictionary into the model
#     clip_model.load_state_dict(clip_state_dict)
#     clip_model.eval()
#     print("Fine-tuned CLIP model loaded successfully.")
#     return clip_model, preprocess

# Load CLIP model
def load_clip_model():
    clip_model_name = 'RN50x4'
    clip_model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_state_dict = torch.load(model_path, map_location=device)
    clip_model.load_state_dict(clip_state_dict["CLIP"])
    return clip_model, preprocess

# Load the Combiner network
def load_combiner():
    feature_dim = 640  # Correct dimension for RN50x4 (adjust if needed)
    projection_dim = 2560
    hidden_dim = 5120
    combiner = Combiner(feature_dim, projection_dim, hidden_dim)
    combiner_state_dict = torch.load(combiner_path, map_location=device)
    combiner.load_state_dict(combiner_state_dict['Combiner'])
    combiner.eval()
    return combiner

# Load the FAISS index and metadata
# def load_faiss_index():
#     try:
#         index = faiss.read_index('image_index.faiss')
#         image_metadata = np.load('image_metadata.npz')
#         image_names = image_metadata['names']
#         print(f"Loaded {len(image_names)} image names.")
#         print(f"FAISS index size: {index.ntotal}")
#         return index, image_names
#     except Exception as e:
#         print(f"Error loading FAISS index or metadata: {e}")
#         raise HTTPException(status_code=500, detail="Error loading FAISS index or metadata")

# Define the preprocess pipeline using targetpad_transform
def create_preprocess_pipeline(target_ratio=1.25):
    return targetpad_transform(target_ratio, clip_model.visual.input_resolution)

# Preprocess the image and return the feature vector
def preprocess_image(image_bytes, preprocess):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        return image_features
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot identify image file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

# Preprocess and encode text
def preprocess_text(text, model):
    if text:
        text_tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        return text_features
    else:
        raise HTTPException(status_code=400, detail="No text provided.")

# Combine image and text features
def combine_features(image_features, text_features, combiner):
    if image_features is not None and text_features is not None:
        combined_features = combiner.combine_features(image_features, text_features)
    elif image_features is not None:
        combined_features = image_features
    elif text_features is not None:
        combined_features = text_features
    else:
        raise HTTPException(status_code=400, detail="Both image and text are missing.")
    return combined_features

# Normalize the query vector for cosine similarity
def normalize_query_vector(query_vector):
    query_vector = query_vector.detach().cpu().numpy().reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_vector)
    return query_vector

# Perform FAISS search to find top k similar images
def search_faiss_index(index, query_vector, k=10):
    try:
        D, I = index.search(query_vector, k)
        return D, I
    except faiss.FaissException as fe:
        raise HTTPException(status_code=500, detail=f"FAISS-specific error: {fe}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during FAISS search: {e}")

# Get the top k ranked images and their similarity scores
def get_top_images(image_names, D, I, k=10):
    top_images = [(image_names[i], D[0][j]) for j, i in enumerate(I[0])]
    return top_images

# Main prediction function
def predict(image_bytes, text, combiner, index, image_names, preprocess, k):
    # Preprocess the image
    image_features = preprocess_image(image_bytes, preprocess)

    # Encode the text
    text_features = preprocess_text(text, clip_model)

    # Combine features
    combined_features = combine_features(image_features, text_features, combiner)

    # Normalize the query vector
    query_vector = normalize_query_vector(combined_features)

    # Search FAISS index
    D, I = search_faiss_index(index, query_vector)

    # Get and return the top images
    top_images = get_top_images(image_names, D, I, k)
    return top_images

# Load the FAISS index and metadata
def load_faiss_index(index_path, metadata_path):
    try:
        index = faiss.read_index(index_path)
        image_metadata = np.load(metadata_path)
        image_names = image_metadata['names']
        print(f"Loaded {len(image_names)} image names.")
        print(f"FAISS index size: {index.ntotal}")
        return index, image_names
    except Exception as e:
        print(f"Error loading FAISS index or metadata: {e}")
        raise HTTPException(status_code=500, detail="Error loading FAISS index or metadata")

# Initialize models and index on startup
@app.on_event("startup")
async def startup():
    global clip_model, preprocess, combiner, index, image_names, fine_tuned_clip_model, fine_tuned_preprocess, fine_tuned_index, fine_tuned_image_names
    clip_model, preprocess = load_clip_model()
    fine_tuned_clip_model, fine_tuned_preprocess = load_fine_tuned_clip_model()
    combiner = load_combiner()
    fine_tuned_index, fine_tuned_image_names = load_faiss_index(fine_tuned_index_path, fine_tuned_metadata_path)
    index, image_names = load_faiss_index('image_index.faiss', 'image_metadata.npz')
    print("Models and index loaded successfully.")

# Define the API endpoint for prediction
@app.post("/predict/")
async def predict_endpoint(
    file: Optional[UploadFile] = File(None),  # Make file optional
    text: Optional[str] = Form(None),         # Make text optional
    k: int = Form(10)  # Default to top 10 results
):
    try:
        # Validate input
        if not file and not text:
            raise HTTPException(status_code=400, detail="At least one input is required: file (image) or text.")

        # Log that the endpoint is hit
        print("Endpoint '/predict/' hit")
        print(f"Received text input: {text}")

        if text and not file:
            print(f"Received text input only: {text}")
            # Encode text
            text_features = preprocess_text(text, fine_tuned_clip_model)
            # Normalize the query vector
            query_vector = normalize_query_vector(text_features)
            # Search FAISS index
            D, I = search_faiss_index(fine_tuned_index, query_vector)
            print("FAISS index searched for text query.")
            # Retrieve top images
            top_images = get_top_images(fine_tuned_image_names, D, I, k=k)
            print("Top images retrieved for text query.")
            return {"top_images": [(name, float(score)) for name, score in top_images]}
        
        else: 
            try:
                # Read the uploaded file
                image_bytes = await file.read()
                print(f"File received: {file.filename}, size: {len(image_bytes)} bytes")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {e}")

            # Perform prediction
            if text.strip():
                # Process both image and text
                top_images = predict(image_bytes, text, combiner, index, image_names, preprocess, k=k)
            else:
                # Process only the image
                print("Text is empty, processing only the image.")
                # Preprocess the image
                image_features = preprocess_image(image_bytes, preprocess)
                print("Image features extracted.")

                # Normalize the query vector
                query_vector = normalize_query_vector(image_features)
                print("Query vector normalized.")

                # Search FAISS index for the top images
                D, I = search_faiss_index(index, query_vector)
                print("FAISS index searched.")

                # Retrieve the top images
                top_images = get_top_images(image_names, D, I, k=k)
                print("Top images retrieved.")

            # Convert numpy.float32 to float
            top_images = [(name, float(score)) for name, score in top_images]
            print("Prediction successful")
            return {"top_images": top_images}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
