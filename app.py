import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load the trained model
@st.cache_resource  # Cache the model to load it only once
def load_model():
    model = tf.keras.models.load_model('skin_cancer_detection_model.h5')
    return model

model = load_model()

# Define classes based on the HAM10000 dataset
classes = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular Lesion']

# Function to make predictions
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image to match model input
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Streamlit App Interface
st.title("Skin Cancer Detection App")
st.write("Upload an image of a skin lesion to get a prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded image to a temporary file
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image("temp_image.jpg", caption="Uploaded Image", use_column_width=True)
    
    # Run prediction
    predicted_class, confidence = predict_image("temp_image.jpg")
    
    # Display results
    st.write(f"### Predicted Class: {predicted_class}")
    st.write(f"### Confidence: {confidence * 100:.2f}%")
