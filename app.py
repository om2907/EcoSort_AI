import streamlit as st
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

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
        categories = ['Organic', 'Recyclable']

        st.image(img, caption='🖼️ Uploaded Image', width=300)
        st.write(f'### 🏷 Waste Type: **{categories[np.argmax(score)]}** (Confidence: {np.max(score) * 100:.2f}%)')

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
