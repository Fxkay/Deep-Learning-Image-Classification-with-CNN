import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input  # ✅ Added

# CIFAR-10 class labels
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Loading the trained model


@st.cache_resource
def load_cnn_model():
    return load_model("cifar10_resnet_final_2.h5")


model = load_cnn_model()

# Title and instructions
st.title("🚀 CIFAR-10 Image Classifier (using - ResNet)")
st.write("Upload an image, and the model will predict the object class.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a PNG or JPG image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Resize to match ResNet input
    image_resized = image.resize((75, 75))

    # ✅ Preprocess using ResNet  preprocessing
    image_array = img_to_array(image_resized)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Output
    st.markdown(f"### 🧠 Predicted Class: **{CLASS_NAMES[class_index]}**")
    st.markdown(f"📊 Confidence Score: `{confidence:.2%}`")
