import streamlit as st
from PIL import Image
from utils import load_trained_model, predict_image


@st.cache_resource
def get_model():
    return load_trained_model()


model, device = get_model()

st.title("Vehicle Image Classifier")
st.write("Upload an image of a vehicle to detect its class.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    predicted_class, confidence = predict_image(model, device, image)
    st.success(f"Predicted class: **{predicted_class}** (confidence: {confidence:.1f}%)")