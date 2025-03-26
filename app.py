import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from PIL import Image
import time

# Hugging Face Model Repository Details
REPO_ID = "omchaudhari2644/Image_classify.keras"  # Replace with your Hugging Face repo
FILENAME = "Image_classify.keras"  # Ensure this file is uploaded to Hugging Face

# Download the model from Hugging Face
st.write("📥 Downloading model from Hugging Face (This may take some time)...")
MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
st.write("✅ Model download complete!")

# Load the model
try:
    model = load_model(MODEL_PATH, compile=False)
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Define categories
data_cat = ['Organic', 'Recyclable']

# Define image dimensions based on the model's expected input shape
img_height = 180  # Height defined in the model
img_width = 180   # Width defined in the model

# Set header and description
st.markdown(
    "<h1 style='text-align: center; color: white; font-size: 40px; font-weight: bold;'>AI-Powered Waste Classification Using KERAS</h1>",
    unsafe_allow_html=True
)

# Function to convert RGBA to RGB
def convert_rgba_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image

# Function to make predictions on an image
def predict_frame(image):
    image = convert_rgba_to_rgb(image)  # Convert RGBA to RGB if needed
    image = image.resize((img_width, img_height))
    img_arr = tf.keras.utils.img_to_array(image)[:, :, :3]  # Ensure only 3 channels (RGB)
    img_bat = tf.expand_dims(img_arr, axis=0)

    # Make predictions
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Get prediction and confidence
    category = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100  # Convert to percentage
    return category, confidence

# Upload image functionality
uploaded_file = st.file_uploader("Upload an image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    uploaded_image = Image.open(uploaded_file)

    # Create columns for layout
    col1, col2 = st.columns(2)

    # Display uploaded image in the first column
    with col1:
        st.image(uploaded_image, caption='Uploaded Image', use_container_width=True)

    # Make predictions on the uploaded image and display in the second column
    with col2:
        category, confidence = predict_frame(uploaded_image)
        st.markdown(f"<h2 style='text-align: center;'><strong>Prediction: {category}<strong></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)

# Run the app
if st.button("Start Webcam", key="start_webcam"):
    st.warning("Webcam feature is not available in hosted environments like Streamlit Cloud.")
