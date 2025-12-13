import cv2
import numpy as np
import streamlit as st  
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions)
from PIL import Image

def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image) :
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
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

def main():
    st.set_page_config(page_title="Image Classifier", layout="centered")
    st.title("Image Classifier using MobileNetV2")
    st.write("Upload an image to classify lets Ai tell you what is in it.")

    @st.cache_resource
    def load_model_cached():
        return load_model()
    
    model = load_model_cached()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        btn = st.button("Classify Image")
        if btn:
            with st.spinner('Classifying...'):
                image = image.open(uploaded_file)
                predictions = classify_image(image)

                if predictions:
                    for _,label, prob in predictions:
                        st.write(f"**{label}**: {prob:.2%}")


if __name__ == "__main__":
    main()