import os
import gdown
import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "Image_classify.keras"
DRIVE_FILE_ID = "1KV2Ea9flhhwPUEaWuU410mEqAhmW6_mE"  # 🔹 Replace with your actual file ID
DOWNLOAD_URL = f"https://drive.google.com/file/d/1KV2Ea9flhhwPUEaWuU410mEqAhmW6_mE/view?usp=sharing={DRIVE_FILE_ID}"

# Function to download the model if missing
def download_model():
    st.write("📥 Downloading model from Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    st.write("✅ Model download complete!")

# Check if the model exists, otherwise download it
if not os.path.exists(MODEL_PATH):
    download_model()

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()


# Check if the model file exists, otherwise download it
if not os.path.exists(MODEL_PATH):
    download_model()

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Define categories
data_cat = ['O', 'R']

# Define image dimensions
img_height = 180
img_width = 180

st.header("EcoSort AI")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess the image
        image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image_load)
        img_bat = tf.expand_dims(img_arr, axis=0)

        # Make predictions
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict[0])  # Apply softmax correctly

        # Display the image
        st.image(image_load, caption='Uploaded Image', width=300)

        # Show prediction results
        st.write(f'### Waste in image is **{data_cat[np.argmax(score)]}** with accuracy of **{np.max(score) * 100:.2f}%**')

        # Show all category scores
        st.write("### Category Confidence:")
        for i, category in enumerate(data_cat):
            st.write(f"{category}: {score[i] * 100:.2f}%")

    except Exception as e:
        st.error(f"Error processing image: {e}")
