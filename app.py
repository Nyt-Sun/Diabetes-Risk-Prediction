import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide",
    page_icon="⚕️"
)

# =============================
# LOAD ARTIFACTS
# =============================
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
median_values = pickle.load(open("model/medians.pkl", "rb"))

# =============================
# UI
# =============================
st.title("⚕️ Diabetes Risk Predictor")

# =============================
# INPUTS
# =============================
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    insulin = st.number_input("Insulin", 0, 900, 80)
    age = st.number_input("Age", 1, 100, 30)

with col2:
    glucose = st.number_input("Glucose", 0, 200, 120)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

# =============================
# PREPROCESSING (ALIGNED)
# =============================
def preprocess_input(data):
    df = pd.DataFrame([data], columns=[
        "Pregnancies", "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age"
    ])

    cols_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

    df = df.fillna(median_values)

    return df.values

# =============================
# PREDICTION
# =============================
if st.button("🔍 RUN ANALYSIS"):

    features = preprocess_input([
        pregnancies, glucose, bp, skin,
        insulin, bmi, dpf, age
    ])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"🚨 HIGH RISK ({prob:.2f}%)")
    else:
        st.success(f"✅ LOW RISK ({prob:.2f}%)")

# =============================
# SIMPLE CHART
# =============================
st.subheader("Key Indicators")

st.bar_chart(pd.DataFrame({
    "Feature": ["Pregnancies","Glucose","BP","Skin","Insulin","BMI","DPF","Age"],
    "Value": [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
}).set_index("Feature"))