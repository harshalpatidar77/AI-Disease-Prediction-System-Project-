import pandas as pd 
def load_data(path):
    return pd.read_csv(path) 
def remove_duplicates(df):
    return df.drop_duplicates() 
def create_features(df):
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
    for column in symptom_columns:
        df[column] = df[column].map({
            "Yes": 1,
            "No": 0
        })
    df["symptom_count"] = df[symptom_columns].sum(axis=1) 
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 40, 60, 100],
        labels=[
            "Young",
            "Adult",
            "Middle_Age",
            "Senior"
        ]
    ) 
    df["blood_pressure_category"] = pd.cut(
        df["blood_pressure"],
        bins=[0, 120, 140, 200],
        labels=[
            "Normal",
            "Elevated",
            "High"
        ]
    ) 
    df["glucose_category"] = pd.cut(
        df["glucose"],
        bins=[0, 100, 140, 300],
        labels=[
            "Normal",
            "Elevated",
            "High"
        ]
    ) 
    return df  