import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from numpy import argmax

# Login credentials
actual_email = "admin"
actual_password = "123"

# Login form
email = st.text_input("Email")
password = st.text_input("Password", type="password")
submit = st.button("Login")

if email == actual_email and password == actual_password:
    st.success("Login successful")
    
    # Load the model
    model = tf.keras.models.load_model('blood.h5')

    st.title("Blood Cell")
    st.write("BLOOD CELL")

    # File upload
    file = st.file_uploader("Upload an image file", type=["jpg", "png"])

    if file is not None:
        # Process the image
        size = (65, 65)
        test_image_o = ImageOps.fit(Image.open(file), size, Image.LANCZOS)
        test_data = np.array(test_image_o, dtype="float") / 255.0
        test_data = test_data.reshape([-1, 65, 65, 3])
        categories = ['Acanthocyte', 'Elliptocyte', 'Macrocyte', 'Microcyte', 'Spherocyte', 'Stomatocyte']

        # Make prediction
        pred = model.predict(test_data)
        max_prob = np.max(pred)  # Highest probability
        predictions = argmax(pred, axis=1)
        confidence_threshold = 0.5 # Set an appropriate threshold
        
        
   
       

        # Check if the model's confidence meets the threshold
        if max_prob < confidence_threshold:
            st.warning("The uploaded image may not be a valid blood cell image. Please upload a clear image of a blood cell.")
        else:
            # Display prediction and additional information
            st.write('Type of cell present:', categories[predictions[0]])
            st.write("The cell may be associated with: \n")
            if categories[predictions[0]] == "Acanthocyte":
                st.write("Abetalipoproteinemia")
                st.write("Liver Disease")
                st.write("Post-Splenectomy")
            elif categories[predictions[0]] == "Elliptocyte":
                st.write("Hereditary Elliptocytosis")
                st.write("Severe iron Deficiency anemia")
            elif categories[predictions[0]] == "Macrocyte":
                st.write("Vitamin B12 Deficiency")
                st.write("MDS")
                st.write("Chemotherapy")
            elif categories[predictions[0]] == "Microcyte":
                st.write("Thalassemia")
                st.write("Pyridoxine Deficiency")
                st.write("Chronic disease anemia")
            elif categories[predictions[0]] == "Spherocyte":
                st.write("Hereditary Spherocytosis")
                st.write("Autoimmune hemolytic anemia")
            elif categories[predictions[0]] == "Stomatocyte":
                st.write("Hereditary Stomatocytosis")
                st.write("Liver disease")
            else:
                st.write("None")

else: 
    st.error("Login Failed")
