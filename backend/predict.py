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
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import torch
from pydantic import BaseModel
from typing import List, Dict, Optional
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# Helper function for resizing images
def resize_image(in_image: Image.Image, target_ratio: float, dim: int) -> Image.Image:
    w, h = in_image.size
    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio < target_ratio:
        out_image = in_image
    else:
        scaled_max_wh = max(w, h) / target_ratio
        hp = max((scaled_max_wh - w) // 2, 0)
        vp = max((scaled_max_wh - h) // 2, 0)
        padding = (int(hp), int(vp), int(hp), int(vp))
        out_image = ImageOps.expand(in_image, border=padding, fill=0)
    out_image = out_image.resize((dim, dim), Image.Resampling.LANCZOS)
    return out_image

def create_faiss_index(image_folder, clip_model, preprocess, index_path, metadata_path):
    """
    Creates a FAISS index and metadata for the images in the specified folder.

    Args:
        image_folder (str): Path to the folder containing images.
        clip_model: Preloaded CLIP model.
        preprocess: Preprocessing function for the CLIP model.
        index_path (str): Path to save the FAISS index.
        metadata_path (str): Path to save the metadata.

    Returns:
        index: Trained FAISS index.
        image_names: List of image names indexed.
    """
    image_names = []
    image_features = []

    # Process images
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path) and not image_name.startswith('._'):
            try:
                # Preprocess and vectorize the image
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_feature = clip_model.encode_image(image).cpu().numpy()
                image_names.append(image_name)
                image_features.append(image_feature)
            except UnidentifiedImageError:
                print(f"Skipping invalid image file: {image_name}")
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

    # Check if any valid images were processed
    if not image_features:
        raise HTTPException(status_code=400, detail="No valid images found for indexing.")

    # Convert features to NumPy array
    image_features = np.vstack(image_features).astype(np.float32)

    # Normalize image features for cosine similarity
    faiss.normalize_L2(image_features)

    # Dynamically determine number of clusters (nlist) for IVF
    dim = image_features.shape[1]
    nlist = min(len(image_features) // 10, 20)  # Adjust number of clusters dynamically

    # Debug: Validate image features
    assert image_features.ndim == 2, f"Expected 2D array, got {image_features.ndim}D"
    assert not np.any(np.isnan(image_features)), "NaN values detected in image features"
    assert image_features.shape[0] >= nlist, f"Number of training points is less than nlist"

    # Create and train FAISS index
    quantizer = faiss.IndexFlatIP(dim)  # Inner Product quantizer for cosine similarity
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(image_features)
    index.add(image_features)

    # Save the FAISS index
    try:
        faiss.write_index(index, index_path)
        print(f"FAISS index saved successfully to '{index_path}'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving FAISS index: {e}")

    # Save metadata
    try:
        np.savez(metadata_path, names=np.array(image_names))
        print(f"Metadata saved successfully to '{metadata_path}'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving metadata: {e}")

    return index, image_names

# Define the image directory path
IMAGE_DIR = "../backend/FashionIQ/resized_images"

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

# Allow CORS for specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

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
    # except faiss.FaissException as fe:
    #     raise HTTPException(status_code=500, detail=f"FAISS-specific error: {fe}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during FAISS search: {e}")

# Get the top k ranked images and their similarity scores
def get_top_images(image_names, D, I, k=10):
    top_images = [(image_names[i], D[0][j]) for j, i in enumerate(I[0][:k])]
    return top_images

def compute_combined_query_vector(image_bytes, text, combiner, index, image_names, preprocess, k):
    # Preprocess the image
    image_features = preprocess_image(image_bytes, preprocess)

    # Encode the text
    text_features = preprocess_text(text, clip_model)

    # Combine features
    combined_features = combine_features(image_features, text_features, combiner)

    # Normalize the query vector
    query_vector = normalize_query_vector(combined_features)

    return query_vector

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
    D, I = search_faiss_index(index, query_vector, k)

    # Get and return the top images
    top_images = get_top_images(image_names, D, I, k)
    return top_images

# Load the FAISS index and metadata
# def load_faiss_index(index_path, metadata_path):
#     try:
#         index = faiss.read_index(index_path)
#         image_metadata = np.load(metadata_path)
#         image_names = image_metadata['names']
#         print(f"Loaded {len(image_names)} image names.")
#         print(f"FAISS index size: {index.ntotal}")
#         return index, image_names
#     except Exception as e:
#         print(f"Error loading FAISS index or metadata: {e}")
#         raise HTTPException(status_code=500, detail="Error loading FAISS index or metadata")
def load_faiss_index(index_path, metadata_path):
    try:
        # Load the FAISS index
        index = faiss.read_index(index_path)

        # Enable the direct map
        index.make_direct_map()

        # Load metadata
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
    global clip_model, preprocess, combiner, index, image_names, fine_tuned_clip_model, fine_tuned_preprocess, fine_tuned_index, fine_tuned_image_names, directory, datasets, fine_tuned_datasets
    datasets = {}
    fine_tuned_datasets = {}
    directory = "../backend/FashionIQ/resized_images"
    clip_model, preprocess = load_clip_model()
    fine_tuned_clip_model, fine_tuned_preprocess = load_fine_tuned_clip_model()
    combiner = load_combiner()
    fine_tuned_index, fine_tuned_image_names = load_faiss_index(fine_tuned_index_path, fine_tuned_metadata_path)
    index, image_names = load_faiss_index('image_index.faiss', 'image_metadata.npz')
    datasets["FashionIQ"] = {"index": index, "names": image_names}
    fine_tuned_datasets["FashionIQ"] = {"index": fine_tuned_index, "names": fine_tuned_image_names}
    print("Models and index loaded successfully.")

from typing import Union, Optional

# Define the API endpoint for prediction
@app.post("/predict/")
async def predict_endpoint(
    file: Optional[UploadFile] = File(None),  # Make file optional
    text: Optional[str] = Form(None),         # Make text optional
    k: Union[str, int] = Form(10),            # Allow k to be string or int, default to 10
    dataset: str = Form(...)                  # Dataset name is required
):
    # Parse k to an integer if it's a string
    try:
        k = int(k)
    except ValueError:
        raise HTTPException(status_code=400, detail="Parameter 'k' must be an integer or a string that can be converted to an integer.")
    
    print(file, text, k, dataset)
    # Ensure the dataset exists
    if dataset not in datasets or dataset not in fine_tuned_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")

    # Retrieve the correct FAISS indices and metadata
    global fine_tuned_index, fine_tuned_image_names, index, image_names

    fine_tuned_index = fine_tuned_datasets[dataset]["index"]  # Correct: Assign FAISS index
    fine_tuned_image_names = fine_tuned_datasets[dataset]["names"]  # Assign names
    index = datasets[dataset]["index"]  # Correct: Assign FAISS index
    image_names = datasets[dataset]["names"]  # Assign names

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
            D, I = search_faiss_index(fine_tuned_index, query_vector, k)
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
                D, I = search_faiss_index(index, query_vector, k)
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
    
def construct_query_vector(current_query, feedback, index, image_names):
    alpha = 1.0  # Tunable weight for the initial query
    beta = 0.75  # Tunable weight for relevant documents
    gamma = 0.25  # Tunable weight for irrelevant documents

    relevant_images = feedback.selected
    irrelevant_images = feedback.unselected

    # Retrieve relevant and irrelevant image features
    def get_image_features(image_list):
        features = []
        for image_name in image_list:
            # Assume FAISS index metadata maps image names to their indices
            index_position = np.where(image_names == image_name)[0]
            if len(index_position) > 0:
                # Fetch the stored feature vector for this image
                features.append(index.reconstruct(int(index_position[0])))
        return np.array(features)

    Dr_features = get_image_features(relevant_images)  # Relevant features
    Dn_features = get_image_features(irrelevant_images)  # Irrelevant features

    # Start with the initial query
    new_query_features = alpha * current_query

    # Add relevant features if any exist
    if Dr_features.size > 0:
        Dr_mean = np.mean(Dr_features, axis=0)  # Compute the centroid
        new_query_features += beta * Dr_mean

    # Subtract irrelevant features if any exist
    if Dn_features.size > 0:
        Dn_mean = np.mean(Dn_features, axis=0)  # Compute the centroid
        new_query_features -= gamma * Dn_mean

    # Normalize the final query vector
    faiss.normalize_L2(new_query_features)

    return new_query_features

class FeedbackIteration(BaseModel):
    selected: List[str]
    unselected: List[str]

class CurrentQuery(BaseModel):
    # keyword: Optional[str]
    # imageFile: Optional[str]  # File is sent separately, so it's optional
    feedback: List[FeedbackIteration]

@app.post("/feedback/")
async def feedback_endpoint(
    k: int = Form(...),
    imageFile: Optional[UploadFile] = File(None),
    keyword: Optional[str] = Form(""),
    current_query: str = Form(...)
):
    try:
        # Parse `current_query` JSON string into a dictionary
        current_query_dict = json.loads(current_query)
        current_query = CurrentQuery(**current_query_dict)

        # Get feedback list
        feedback_list = current_query.feedback

        # Initialize query_vector
        textOnly = False
        query_vector = None

        # Process text-only input
        if keyword and not imageFile:
            textOnly = True
            print(f"Received text input only: {keyword}")
            text_features = preprocess_text(keyword, fine_tuned_clip_model)
            query_vector = normalize_query_vector(text_features)

        # Process image input
        elif imageFile:
            print(f"Received image file: {imageFile.filename}")
            image_bytes = await imageFile.read()
            image_features = preprocess_image(image_bytes, preprocess)
            query_vector = normalize_query_vector(image_features)

            # Combine with text if present
            if keyword:
                combined_features = combine_features(
                    image_features,
                    preprocess_text(keyword, clip_model),
                    combiner
                )
                query_vector = normalize_query_vector(combined_features)

        # Apply feedback iterations
        for feedback_iteration in feedback_list:
            if textOnly:
                query_vector = construct_query_vector(
                    query_vector,
                    feedback_iteration,
                    fine_tuned_index,
                    fine_tuned_image_names
                )
            else:
                query_vector = construct_query_vector(
                    query_vector,
                    feedback_iteration,
                    index,
                    image_names
                )

        # Perform search with the revised query vector
        if textOnly:
            D, I = search_faiss_index(fine_tuned_index, query_vector, k)
            updated_results = get_top_images(fine_tuned_image_names, D, I, k)
        else:
            D, I = search_faiss_index(index, query_vector, k)
            updated_results = get_top_images(image_names, D, I, k)

        return {"updated_results": [(name, float(score)) for name, score in updated_results]}

    except Exception as e:
        print(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Route to create a new dataset
@app.post("/create_dataset/")
async def create_dataset(
    dataset_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # dataset_dir = os.path.join("backend", dataset_name)
    dataset_dir = dataset_name
    resized_images_dir = os.path.join(dataset_dir, "resized_images")

    # Create directories for the dataset
    os.makedirs(resized_images_dir, exist_ok=True)

    # Resize and save uploaded images
    for file in files:
        try:
            image = Image.open(BytesIO(await file.read()))
            resized_image = resize_image(image, target_ratio=1.0, dim=224)
            resized_image.save(os.path.join(resized_images_dir, file.filename))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a valid image.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image {file.filename}: {e}")

    # Create FAISS index and metadata for the base model
    index_path = os.path.join(dataset_dir, f"image_index_{dataset_name}.faiss")
    metadata_path = os.path.join(dataset_dir, f"image_metadata_{dataset_name}.npz")
    index, image_names = create_faiss_index(resized_images_dir, clip_model, preprocess, index_path, metadata_path)

    # Create FAISS index and metadata for the fine-tuned model
    fine_tuned_index_path = os.path.join(dataset_dir, f"fine_tuned_image_index_{dataset_name}.faiss")
    fine_tuned_metadata_path = os.path.join(dataset_dir, f"fine_tuned_image_metadata_{dataset_name}.npz")
    fine_tuned_index, fine_tuned_image_names = create_faiss_index(
        resized_images_dir, fine_tuned_clip_model, fine_tuned_preprocess, fine_tuned_index_path, fine_tuned_metadata_path
    )

    # Update the global dataset dictionaries
    global datasets, fine_tuned_datasets, directory
    datasets[dataset_name] = {"index": index, "names": image_names}
    fine_tuned_datasets[dataset_name] = {"index": fine_tuned_index, "names": fine_tuned_image_names}
    directory = f"../backend/{dataset_name}/resized_images"

    return {"message": f"Dataset '{dataset_name}' created successfully.", "image_count": len(image_names)}
    
from fastapi.responses import FileResponse

@app.get("/images/{dataset_name}/{file_path:path}")
async def serve_image(dataset_name: str, file_path: str):
    """
    Serve static files dynamically based on the dataset.
    """
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    image_dir = f"../backend/{dataset_name}/resized_images"
    file_full_path = os.path.join(image_dir, file_path)

    if not os.path.isfile(file_full_path):
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(file_full_path)

# Route to append images to an existing dataset
@app.post("/append_to_dataset/")
async def append_to_dataset(
    dataset_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    global datasets, fine_tuned_datasets

    print(f"Dataset received: {dataset_name}")
    print(f"Available datasets: {datasets.keys()}")
    print(f"Available fine-tuned datasets: {fine_tuned_datasets.keys()}")

    # Ensure the dataset exists
    if dataset_name not in datasets or dataset_name not in fine_tuned_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")

    print("Dataset exists, proceeding with appending images.")
    if dataset_name != "FashionIQ":
        dataset_dir = dataset_name
        resized_images_dir = os.path.join(dataset_dir, "resized_images")
        index_path = os.path.join(dataset_dir, f"image_index_{dataset_name}.faiss")
        metadata_path = os.path.join(dataset_dir, f"image_metadata_{dataset_name}.npz")
        fine_tuned_index_path = os.path.join(dataset_dir, f"fine_tuned_image_index_{dataset_name}.faiss")
        fine_tuned_metadata_path = os.path.join(dataset_dir, f"fine_tuned_image_metadata_{dataset_name}.npz")
    else:
        resized_images_dir = f"FashionIQ/resized_images"
        index_path = f"image_index.faiss"
        metadata_path = f"image_metadata.npz"
        fine_tuned_index_path = f"fine_tuned_image_index.faiss"
        fine_tuned_metadata_path = f"fine_tuned_image_metadata.npz"

    print(f"Resized images directory: {resized_images_dir}")
    print(f"Index paths: {index_path}, {fine_tuned_index_path}")

    # Resize and save uploaded images
    new_image_names = []
    for file in files:
        try:
            image = Image.open(BytesIO(await file.read()))
            resized_image = resize_image(image, target_ratio=1.0, dim=224)
            resized_image.save(os.path.join(resized_images_dir, file.filename))
            new_image_names.append(file.filename)
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a valid image.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image {file.filename}: {e}")

    print(f"New images resized and saved: {new_image_names}")

    # Load existing index and metadata
    try:
        existing_index, existing_image_names = load_faiss_index(index_path, metadata_path)
        fine_tuned_existing_index, fine_tuned_existing_image_names = load_faiss_index(
            fine_tuned_index_path, fine_tuned_metadata_path
        )
        print("Loaded existing indices and metadata.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading existing indices: {e}")

    # Vectorize new images and append to the base model index
    try:
        print("Starting vectorization of new images...")
        new_image_features = vectorize_new_images(resized_images_dir, new_image_names, clip_model, preprocess)
        print(f"Vectorization complete. New image features shape: {new_image_features.shape}")
        
        print(f"Existing FAISS index dimensions: {existing_index.d}")
        if new_image_features.shape[1] != existing_index.d:
            raise ValueError(f"Dimension mismatch: new vectors have {new_image_features.shape[1]} dims, expected {existing_index.d}.")
        
        print(f"Adding {new_image_features.shape[0]} new vectors to the FAISS index...")
        existing_index.add(new_image_features)
        print("New vectors successfully added to the FAISS index.")
        
        # Ensure consistent data type for updated_image_names
        updated_image_names = np.concatenate([existing_image_names, np.array(new_image_names)])
        print(f"Updated image names count: {len(updated_image_names)}")
        print("Base model index update complete.")
    except ValueError as ve:
        print(f"Dimension mismatch error: {ve}")
        raise HTTPException(status_code=500, detail=f"Dimension mismatch error: {ve}")
    except Exception as e:
        print(f"Error during base model index update: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating base model index: {e}")

    # Save updated base model index and metadata
    try:
        faiss.write_index(existing_index, index_path)
        np.savez(metadata_path, names=np.array(updated_image_names))
        print("Base model index and metadata saved.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving base model index: {e}")

    # Vectorize new images and append to the fine-tuned model index
    try:
        fine_tuned_new_image_features = vectorize_new_images(
            resized_images_dir, new_image_names, fine_tuned_clip_model, fine_tuned_preprocess
        )
        if fine_tuned_new_image_features.shape[1] != fine_tuned_existing_index.d:
            raise ValueError(f"Dimension mismatch: fine-tuned vectors have {fine_tuned_new_image_features.shape[1]} dims, expected {fine_tuned_existing_index.d}.")
        fine_tuned_existing_index.add(fine_tuned_new_image_features)
        fine_tuned_updated_image_names = np.concatenate([fine_tuned_existing_image_names, np.array(new_image_names)])
        print("Fine-tuned model index updated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating fine-tuned model index: {e}")

    # Save updated fine-tuned model index and metadata
    try:
        faiss.write_index(fine_tuned_existing_index, fine_tuned_index_path)
        np.savez(fine_tuned_metadata_path, names=np.array(fine_tuned_updated_image_names))
        print("Fine-tuned model index and metadata saved.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving fine-tuned model index: {e}")

    # Update the global dataset dictionaries
    try:
        datasets[dataset_name] = {"index": existing_index, "names": updated_image_names}
        fine_tuned_datasets[dataset_name] = {
            "index": fine_tuned_existing_index,
            "names": fine_tuned_updated_image_names,
        }
        print("Global dataset dictionaries updated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating global dataset dictionaries: {e}")

    return {
        "message": f"Successfully appended {len(new_image_names)} images to dataset '{dataset_name}'.",
        "new_image_count": len(new_image_names),
        "total_image_count": len(updated_image_names),
    }

# Helper function to vectorize new images
def vectorize_new_images(image_folder, new_image_names, clip_model, preprocess):
    """
    Vectorizes new images by preprocessing and encoding them with the CLIP model.

    Args:
        image_folder (str): Folder containing the new images.
        new_image_names (list): List of new image filenames to process.
        clip_model: Preloaded CLIP model.
        preprocess: Preprocessing function for the CLIP model.

    Returns:
        np.ndarray: Array of vectorized image features.
    """
    new_image_features = []

    for image_name in new_image_names:
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path):
            try:
                # Preprocess and vectorize the image
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_feature = clip_model.encode_image(image).cpu().numpy()
                new_image_features.append(image_feature)
            except UnidentifiedImageError:
                print(f"Skipping invalid image file: {image_name}")
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

    # If no valid images are found, raise an error
    if not new_image_features:
        raise ValueError("No valid images were processed for vectorization.")

    # Convert features to NumPy array and normalize
    new_image_features = np.vstack(new_image_features).astype(np.float32)
    faiss.normalize_L2(new_image_features)

    return new_image_features

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
