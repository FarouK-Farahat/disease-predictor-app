import streamlit as st
import joblib
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("disease_predictor_model.joblib")

model = load_model()

# Streamlit UI setup
st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Symptom-Based Disease Predictor")
st.markdown("Enter your symptoms below, separated by commas.\n\n_Example: `fever, cough, sore throat`_")

# Input from user
user_input = st.text_input("Symptoms", "")

if st.button("🔍 Predict Disease"):
    if user_input.strip():
        predicted_label = model.predict([user_input])[0]
        st.success(f"Most Likely Disease: **{predicted_label}**")
    else:
        st.warning("Please enter at least one symptom.")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")