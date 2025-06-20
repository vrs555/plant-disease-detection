# streamlit_app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import pandas as pd

# Load model and metadata
model = load_model("plant_disease_model.h5")

with open("class_indices.json") as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)
for name, index in class_indices.items():
    class_names[index] = name

# 📋 Disease treatment & prevention suggestions
disease_info = {
    "Pepper_bell___Bacterial_spot": "➡ Use certified, disease-free seeds.\n➡ Apply copper-based bactericides.\n➡ Avoid overhead irrigation.",
    "Pepper_bell___healthy": "✅ Your plant is healthy! Keep monitoring and maintain good practices.",
    "Potato___Early_blight": "➡ Use fungicides like mancozeb or chlorothalonil.\n➡ Remove infected debris.\n➡ Rotate crops.",
    "Potato___healthy": "✅ Healthy potato plant! Keep monitoring regularly.",
    "Potato___Late_blight": "➡ Apply metalaxyl-based fungicides.\n➡ Remove affected parts.\n➡ Ensure good drainage.",
    "Tomato_Target_Spot": "➡ Use chlorothalonil or mancozeb sprays.\n➡ Avoid water splash on leaves.",
    "Tomato_Tomato_mosaic_virus": "➡ Destroy infected plants.\n➡ Disinfect tools.\n➡ Avoid tobacco exposure.",
    "Tomato_Tomato_YellowLeaf_Curl_Virus": "➡ Remove infected plants.\n➡ Use virus-resistant seeds.\n➡ Control whiteflies.",
    "Tomato_Bacterial_spot": "➡ Use copper sprays.\n➡ Avoid handling plants when wet.",
    "Tomato_Early_blight": "➡ Apply chlorothalonil-based fungicides.\n➡ Remove infected leaves.\n➡ Improve spacing.",
    "Tomato_healthy": "✅ Tomato plant is healthy! Continue best practices.",
    "Tomato_Late_blight": "➡ Use copper or metalaxyl fungicides.\n➡ Remove infected leaves.\n➡ Avoid wet foliage overnight.",
    "Tomato_Leaf_Mold": "➡ Use sulfur fungicides.\n➡ Improve air circulation.",
    "Tomato_Septoria_leaf_spot": "➡ Use neem oil or fungicides.\n➡ Remove spotted leaves.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "➡ Use neem oil or insecticidal soap.\n➡ Keep humidity high."
}

# UI
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image or take a photo to detect plant disease.")

# Upload or camera input
uploaded_file = st.file_uploader("📁 Upload leaf image", type=["jpg", "jpeg", "png"])
camera_photo = st.camera_input("📸 Or take a photo using your phone")
image_input = camera_photo if camera_photo else uploaded_file

# Prediction logic (runs only once)
if image_input is not None:
    st.image(image_input, caption="🖼️ Uploaded Leaf", use_column_width=True)

    img = image.load_img(image_input, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    pretty_name = predicted_class.replace("___", " - ").replace("_", " ")
    st.success(f"🎯 **Prediction:** {pretty_name}")
    st.info(f"🔬 **Confidence:** {confidence:.2f}%")

    # Show treatment
    if predicted_class in disease_info:
        st.warning("🩺 **Treatment Advice:**\n\n" + disease_info[predicted_class])
    else:
        st.warning("⚠️ No treatment info found for this disease.")

    # Load doctor data
    doctor_df = pd.read_csv("C:/Users/VIGHNESH/Desktop/Vighnesh_related/internship 5th sem/internship_5_project-old/plant_doctors.csv")
    locations = sorted(doctor_df["location"].unique())

    location = st.selectbox("📍 Select your city or town to find nearby plant doctors:", locations)

    if location:
        st.subheader("👨‍⚕️ Nearby Plant Health Experts")
        matches = doctor_df[doctor_df["location"] == location]
        if not matches.empty:
            for _, row in matches.iterrows():
                st.markdown(f"""
                **🧑‍🔬 {row['name']}**  
                *{row['designation']}*  
                📞 {row['contact']}  
                """)
        else:
            st.info("❌ No plant doctors found for this location.")
