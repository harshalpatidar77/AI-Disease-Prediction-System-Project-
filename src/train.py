import os 
import joblib 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.ensemble import RandomForestClassifier  
DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/best_model.pkl" 
def train_model(): 
    df = pd.read_csv(DATA_PATH) 
    X = df.drop("disease", axis=1)
    y = df["disease"] 
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist() 
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist() 
    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ]),
            categorical_features
        )
    ]) 
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ) 
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ]) 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    ) 
    pipeline.fit(X_train, y_train) 
    os.makedirs("models", exist_ok=True) 
    joblib.dump(
        pipeline,
        MODEL_PATH
    ) 
    print("Model trained successfully.")
    print(f"Model saved at: {MODEL_PATH}")
if __name__ == "__main__":
    train_model()

