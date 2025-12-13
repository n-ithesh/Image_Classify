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