import streamlit as st
import requests
from PIL import Image
import os

# Backend API URL
API_URL = "http://localhost:8000/predict/"
IMAGE_BASE_PATH = "../backend/FashionIQ/resized_images/"

# Title and description
st.title("Fashion Image Retrieval")
st.write("Upload an image and optionally enter text to retrieve the most relevant fashion images.")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Display image preview
if uploaded_file:
    try:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        # Display the image
        st.image(image, caption="Uploaded Image", use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying the image preview: {e}")

# Text input
text_input = st.text_input("Describe your fashion need (optional):")

if st.button("Send"):
    if uploaded_file:
        with st.spinner("Fetching relevant images..."):
            try:
                # Read the uploaded file as bytes only once
                image_bytes = uploaded_file.getvalue()

                # Prepare the file and text for the POST request
                files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                data = {"text": text_input}

                # Send the request to the backend
                response = requests.post(API_URL, files=files, data=data)
                response.raise_for_status()  # Raise an error for failed HTTP responses

                # Parse the JSON response
                results = response.json()

                # Display results
                st.subheader("Search Results")
                for idx, item in enumerate(results["top_images"]):
                    name = item[0]  # Image name
                    score = item[1]  # Similarity score
                    image_path = os.path.join(IMAGE_BASE_PATH, name)

                    if os.path.exists(image_path):
                        with open(image_path, "rb") as img_file:
                            img_data = img_file.read()
                            st.image(img_data, caption=f"Result {idx + 1} (Score: {score:.2f})", use_container_width=True)
                    else:
                        st.warning(f"Image not found: {name}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error contacting backend: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please upload an image before sending.")
