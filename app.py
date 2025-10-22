import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow GPU logs

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# 🧠 PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="🐾 Animal Classifier | AI Vision",
    page_icon="🐾",
    layout="wide"
)

# =========================================================
# 🎨 CUSTOM STYLES
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    color: #F5F5F5;
}

h1 { text-align: center; color: #F5F5F5; font-weight: 700; letter-spacing: 1px; }

.upload-box {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    border: 2px dashed rgba(255,255,255,0.25);
    transition: 0.3s;
}
.upload-box:hover { border-color: #8E2DE2; box-shadow: 0 0 25px rgba(142,45,226,0.4); }

.image-card, .prediction-card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(31,38,135,0.37);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(10px);
    animation: fadeIn 1.2s ease;
    text-align: center;
}
.prediction-card h2 { color: #A29BFE; font-weight: 600; }
.prediction-card h1 { color: #00FFA3; font-size: 2.2em; font-weight: 700; }

footer {visibility: hidden;}
@keyframes fadeIn { from {opacity: 0; transform: translateY(20px);} to {opacity: 1; transform: translateY(0);} }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🦁 TITLE & INTRO
# =========================================================
st.markdown("<h1>🐾 Animal Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Upload an animal image — the AI will predict it 🦊</p>", unsafe_allow_html=True)

# =========================================================
# 🧠 LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("animal_classifier_model.keras")

model = load_model()

# =========================================================
# 🐯 CLASS LABELS
# =========================================================
CLASS_NAMES = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
    'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
    'Panda', 'Tiger', 'Zebra'
]

# =========================================================
# 📤 FILE UPLOADER
# =========================================================
st.markdown("<div class='upload-box'>📸 <b>Select an image (JPG, PNG)</b></div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# =========================================================
# 🧩 MAIN CONTENT
# =========================================================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")  # Force RGB
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Preprocess
        # =========================================================
    # 🖼️ Preprocess Image Safely
    # =========================================================
    image = Image.open(uploaded_file)

    # 🔹 Force 3-channel RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 🔹 Resize correctly to 224x224 (EfficientNetB0 standard)
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # ✅ Double-check shape (should be 224x224x3)
    if img_array.shape[-1] != 3:
        st.error(f"Invalid image format! Expected 3 channels but got {img_array.shape[-1]}")
        st.stop()

    # 🔹 Expand and preprocess for EfficientNet
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)


    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    pred_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🧠 Predicted Animal</h2>
            <h1>{pred_class}</h1>
            <p style="margin-top:10px;"><b>Confidence:</b> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(confidence)/100)
        st.caption("🎯 Model: EfficientNetB0 (ImageNet pretrained)")

    st.divider()
    st.success("🎉 Upload another image to classify again!")

else:
    st.info("⬆️ Upload an animal image above to start prediction.")
