import os
import gdown
import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Google Drive model file ID
DRIVE_FILE_ID = "1KV2Ea9flhhwPUEaWuU410mEqAhmW6_mE"
MODEL_PATH = "Image_classify.keras"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

def download_model():
    """Download model if not found."""
    if not os.path.exists(MODEL_PATH):
        st.write("📥 Downloading model from Google Drive (This may take some time)...")
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
        st.write("✅ Model download complete!")

# Ensure model is available before loading
download_model()

# Load the model
try:
    model = load_model(MODEL_PATH, compile=False)
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Streamlit UI
st.header("♻️ EcoSort AI - Waste Classification")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    try:
        img = tf.keras.utils.load_img(uploaded_file, target_size=(180, 180))
        img_arr = tf.keras.utils.img_to_array(img) / 255.0
        img_bat = tf.expand_dims(img_arr, axis=0)

        predictions = model.predict(img_bat)
        score = tf.nn.softmax(predictions[0])
        categories = ['O', 'R']

        st.image(img, caption='🖼️ Uploaded Image', width=300)
        st.write(f'### 🏷 Waste Type: **{categories[np.argmax(score)]}** (Confidence: {np.max(score) * 100:.2f}%)')

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
