import joblib
import pandas as pd
import os
import sys 
sys.path.append(os.path.abspath(".")) 
from src.feature_engineering import create_features
MODEL_PATH = "models/best_model.pkl"
def predict_disease(input_data):
    model = joblib.load(MODEL_PATH)
    # Convert dictionary/list into DataFrame if necessary
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    # Apply the same feature engineering used during training
    input_data = create_features(input_data)
    # Prediction
    prediction = model.predict(input_data)
    # Probability
    probabilities = model.predict_proba(input_data)
    probability_df = pd.DataFrame({
        "Disease": model.classes_,
        "Probability": probabilities[0]
    })
    probability_df["Probability (%)"] = (
        probability_df["Probability"] * 100
    ) 
    return prediction[0], probability_df