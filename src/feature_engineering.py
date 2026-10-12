import pandas as pd 
def create_features(df):
    """
    Create engineered features for the AI Disease Prediction System.

    Input:
        DataFrame containing the original raw features.

    Output:
        DataFrame containing original features + engineered features.
    """
    df = df.copy()
    symptom_columns = [
        "fever",
        "cough",
        "fatigue",
        "headache",
        "chest_pain",
        "breathing_difficulty",
        "nausea"
    ]
    # 1. Symptom Count
    binary_symptoms = df[symptom_columns].apply(
        lambda col: col.map({
            "Yes": 1,
            "No": 0,
            1: 1,
            0: 0
        })
    )
    df["symptom_count"] = binary_symptoms.sum(axis=1)
    # 2. Age Group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 40, 60, 100],
        labels=[
            "Child",
            "Young_Adult",
            "Middle_Aged",
            "Senior"
        ],
        include_lowest=True
    )
    # 3. Blood Pressure Category
    df["blood_pressure_category"] = pd.cut(
        df["blood_pressure"],
        bins=[0, 90, 120, 140, 200],
        labels=[
            "Low",
            "Normal",
            "Elevated",
            "High"
        ],
        include_lowest=True
    )
    # 4. Glucose Category
    df["glucose_category"] = pd.cut(
        df["glucose"],
        bins=[0, 70, 100, 126, 300],
        labels=[
            "Low",
            "Normal",
            "Prediabetic",
            "High"
        ],
        include_lowest=True
    ) 
    return df 