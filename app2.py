import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2 as cv
from PIL import Image

# Load the trained model
model = joblib.load('Random Forest_model.pt')  # Ensure this file exists in your working directory

# Function to process the uploaded image
def image_process(image):
    try:
        # Convert PIL image to grayscale NumPy array
        img_a = np.array(image.convert("L"))
        
        # Resize image to 80x80
        img_r = cv.resize(img_a, (80, 80))
        
        # Flatten the array and convert it to a DataFrame
        img_f = img_r.flatten()
        img_d = pd.DataFrame([img_f])  # Wrap in a list to ensure proper shape
        
        # Predict flower type
        p_value = model.predict(img_d)
        return p_value[0]
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit UI
st.title("IRIS Flower Classification")
st.write("Upload an image of a flower, and the model will classify it.")

# Upload file
file = st.file_uploader("Choose an image", type=['jpg', 'png'])

if file is not None:
    try:
        # Open the image
        image_x = Image.open(file)

        # Display the uploaded image
        st.image(image_x, caption="Uploaded Image", use_column_width=True)
        #st.write("The prediction is here")
        
        # Process and predict
        prediction = image_process(image_x)
        #st.write(prediction)
        if prediction is not None:
            st.write(f"The classification of the flower is: **{prediction}**")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload an image file.")
