import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# 🧠 PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="🐾 Animal Classifier", page_icon="🐾", layout="wide")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #E0C3FC, #8EC5FC);
        color: #333;
        font-family: 'Poppins', sans-serif;
    }
    h1, h3 {
        text-align: center;
        color: #4A148C;
    }
    .stButton>button {
        background-color: #6A1B9A;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #8E24AA;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🐾 AI Animal Image Classifier")
st.subheader("Upload an image to identify the animal 🦁🐘🐶🐱")

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
# 📸 IMAGE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# ...existing code...

# Update the preprocessing section
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Force convert to RGB
    image = image.convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    with col2:
        st.write("🎯 Processing image...")

        # ✅ Preprocess image - ensure correct size and channels
        img = image.resize((224, 224))  # Confirm this matches your model's input size
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

# ...existing code...

        # ✅ Predict
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        pred_class = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.markdown(f"""
            <div style='background-color:#F3E5F5;padding:20px;border-radius:15px;box-shadow:0 0 10px rgba(0,0,0,0.1);'>
                <h3>🧠 Prediction: <b>{pred_class}</b></h3>
                <p style='font-size:18px;'>Confidence: <b>{confidence:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)

        st.progress(float(confidence) / 100)
        st.success("✅ Try uploading another image!")

else:
    st.info("📥 Please upload an animal image to start classification.")
