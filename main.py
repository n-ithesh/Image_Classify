import numpy as np
import streamlit as st  
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions)
from PIL import Image

def load_model():
    return MobileNetV2(weights='imagenet')

def preprocess_image(image):
    image = image.resize((224, 224))   
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return []

@st.cache_resource
def load_model_cached():
    return load_model()

def main():
    st.set_page_config(
        page_title="AI Image Classifier",
        page_icon="🧠",
        layout="centered"
    )

    # ---------- SIDEBAR ----------
    st.sidebar.title("🧠 Image Classifier")
    st.sidebar.write(
        "This app uses **MobileNetV2**, a pre-trained deep learning model, "
        "to identify objects in images."
    )
    st.sidebar.markdown("---")
    st.sidebar.write("**How it works:**")
    st.sidebar.write(
        "1. Upload an image\n"
        "2. Click *Classify Image*\n"
        "3. View top predictions"
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Built   using Streamlit & TensorFlow")

    # ---------- MAIN TITLE ----------
    st.markdown(
        "<h1 style='text-align: center;'>AI Image Classifier</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: grey;'>"
        "Upload an image and let AI tell you what it sees"
        "</p>",
        unsafe_allow_html=True
    )

    st.write("")  # spacing

    # ---------- LOAD MODEL ----------
    model = load_model_cached()

    # ---------- UPLOAD SECTION ----------
    with st.container():
        st.subheader(" Upload Image")

        uploaded_file = st.file_uploader(
            "Supported formats: JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"]
        )

    # ---------- IMAGE PREVIEW ----------
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")

        st.write("")
        st.subheader(" Image Preview")
        st.image(
            pil_image,
            use_container_width=True
        )

        # ---------- CLASSIFY BUTTON ----------
        st.write("")
        if st.button(" Classify Image", use_container_width=True):
            with st.spinner("Analyzing image..."):
                predictions = classify_image(model, pil_image)

            # ---------- RESULTS ----------
            if predictions:
                st.write("")
                st.subheader(" Prediction Results")

                for i, (_, label, prob) in enumerate(predictions, start=1):
                    st.markdown(
                        f"""
                        **{i}. {label.replace('_', ' ').title()}**  
                        Confidence: **{prob:.2%}**
                        """
                    )

if __name__ == "__main__":
    main()
