import streamlit as st
import pandas as pd
import joblib
import os
import sys 
# Add project root to Python path
sys.path.append(os.path.abspath("."))
from src.feature_engineering import create_features
# Page Configuration
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)
# Load Model
MODEL_PATH = "models/best_model.pkl"
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()
# Title
st.title("🩺 AI Disease Prediction System")
st.write(
    "Enter the patient's basic health information and symptoms "
    "to generate an educational disease-risk prediction."
)
st.warning(
    "⚠️ This application is an educational machine-learning "
    "prototype and is not a medical diagnostic tool."
)
# Patient Information
st.header("Patient Information")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )
    temperature = st.number_input(
        "Temperature (°F)",
        min_value=90.0,
        max_value=110.0,
        value=98.6,
        step=0.1
    )
with col2:
    heart_rate = st.number_input(
        "Heart Rate",
        min_value=40,
        max_value=180,
        value=75
    )
    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=60,
        max_value=200,
        value=120
    )
    glucose = st.number_input(
        "Glucose",
        min_value=40,
        max_value=300,
        value=100
    )
with col3:
    cholesterol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=400,
        value=180
    )
# Symptoms
st.header("Symptoms")
col1, col2, col3 = st.columns(3)
with col1:
    fever = st.selectbox(
        "Fever",
        ["No", "Yes"]
    )
    cough = st.selectbox(
        "Cough",
        ["No", "Yes"]
    )
    fatigue = st.selectbox(
        "Fatigue",
        ["No", "Yes"]
    )
with col2:
    headache = st.selectbox(
        "Headache",
        ["No", "Yes"]
    )
    chest_pain = st.selectbox(
        "Chest Pain",
        ["No", "Yes"]
    )
with col3:

    breathing_difficulty = st.selectbox(
        "Breathing Difficulty",
        ["No", "Yes"]
    )

    nausea = st.selectbox(
        "Nausea",
        ["No", "Yes"]
    )
# Prediction
st.divider()
if st.button(
    "🔍 Predict Disease",
    use_container_width=True
): 
    # Create raw input DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "temperature": [temperature],
        "heart_rate": [heart_rate],
        "blood_pressure": [blood_pressure],
        "glucose": [glucose],
        "cholesterol": [cholesterol],
        "fever": [fever],
        "cough": [cough],
        "fatigue": [fatigue],
        "headache": [headache],
        "chest_pain": [chest_pain],
        "breathing_difficulty": [breathing_difficulty],
        "nausea": [nausea]
    })
    # Apply SAME feature engineering used during training
    input_data = create_features(input_data)
    # Prediction
    try:
        prediction = model.predict(input_data)

        probabilities = model.predict_proba(
            input_data
        )
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
        st.stop()
    # Extract prediction
    predicted_disease = prediction[0]
    confidence = probabilities.max() * 100
    # Display Prediction
    st.subheader("Prediction Result")
    st.success(
        f"Predicted Disease: {predicted_disease}"
    )
    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )
    # Probability Distribution
    st.subheader("Prediction Probability")
    probability_df = pd.DataFrame({
        "Disease": model.classes_,
        "Probability": probabilities[0]
    })
    probability_df["Probability (%)"] = (
        probability_df["Probability"] * 100
    )
    probability_df = probability_df.sort_values(
        "Probability",
        ascending=False
    )
    st.dataframe(
        probability_df[
            ["Disease", "Probability (%)"]
        ].reset_index(drop=True),
        use_container_width=True
    )
    st.bar_chart(
        probability_df.set_index("Disease")[
            "Probability (%)"
        ]
    )
    # Display Engineered Features
    with st.expander("View Engineered Features"):

        st.dataframe(
            input_data,
            use_container_width=True
        )
# Footer
st.divider()
st.caption(
    "AI Disease Prediction System | "
    "Machine Learning Classification Project"
) 
st.caption(
    "For educational purposes only. "
    "Do not use this prediction as a substitute for professional medical advice."
)   
