import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
) 
DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/best_model.pkl" 
# Load Dataset 
df = pd.read_csv(DATA_PATH)
X = df.drop("disease", axis=1)
y = df["disease"] 
# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
) 
# Load Saved Model  
model = joblib.load(MODEL_PATH)
# Prediction
predictions = model.predict(X_test)
# Evaluation Metrics
accuracy = accuracy_score(
    y_test,
    predictions
)
precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
) 
recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)
f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)
# Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)
os.makedirs(
    "reports",
    exist_ok=True
)
plt.figure(figsize=(9, 7))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)
disp.plot(
    ax=plt.gca(),
    xticks_rotation=30,
    values_format="d",
    colorbar=False
)
plt.title(
    "AI Disease Prediction - Confusion Matrix"
)
plt.tight_layout()
plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
print("\nConfusion matrix saved to:")
print("reports/confusion_matrix.png") 