import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/diabetes.csv")

# =========================
# HANDLE INVALID ZEROS
# =========================
cols_with_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# PIPELINE (IMPUTER + SCALER + MODEL)
# =========================
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# =========================
# TRAIN
# =========================
pipeline.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# SAVE PIPELINE
# =========================
os.makedirs("model", exist_ok=True)

pickle.dump(pipeline, open("model/pipeline.pkl", "wb"))

print("Pipeline saved successfully!")