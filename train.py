import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/diabetes.csv")

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# SCALING
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
print(f"Model Accuracy: {accuracy:.4f}")

# =========================
# SAVE MODEL + SCALER + DATA
# =========================

import os
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

# Save test data for Streamlit evaluation
pickle.dump(X_test_scaled, open("model/X_test.pkl", "wb"))
pickle.dump(y_test.values, open("model/y_test.pkl", "wb"))

print("Model, scaler, and test data saved successfully!")