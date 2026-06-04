import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/diabetes.csv")

# =========================
# HANDLE MISSING VALUES
# =========================

# In the Pima Indians Diabetes dataset,
# zeros in these columns represent missing values

columns_with_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace invalid zeros with NaN
df[columns_with_missing] = df[columns_with_missing].replace(0, np.nan)

print("\nMissing Values Before Filling:")
print(df.isnull().sum())

# =========================
# FEATURES AND TARGET
# =========================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# FILL MISSING VALUES
# =========================

# Calculate median values from TRAINING DATA ONLY
median_values = X_train.median()

# Fill missing values
X_train = X_train.fillna(median_values)
X_test = X_test.fillna(median_values)

print("\nMissing Values After Filling:")
print(X_train.isnull().sum())

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODEL TRAINING
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test_scaled)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL ARTIFACTS
# =========================

os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

# Save test data for evaluation
pickle.dump(X_test_scaled, open("model/X_test.pkl", "wb"))
pickle.dump(y_test.values, open("model/y_test.pkl", "wb"))

print("\nModel, scaler, and test data saved successfully!")