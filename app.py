import streamlit as st
import pickle
import numpy as np
import pandas as pd

# =============================
# LOAD PIPELINE
# =============================
pipeline = pickle.load(open("model/pipeline.pkl", "rb"))

# =============================
# UI CONFIG
# =============================
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide",
    page_icon="⚕️"
)

st.title("⚕️ Diabetes Risk Predictor (ML Pipeline)")

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
# PREDICTION
# =============================
if st.button("🔍 RUN ANALYSIS"):

    input_data = pd.DataFrame([[
        pregnancies, glucose, bp, skin,
        insulin, bmi, dpf, age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    prediction = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(f"🚨 HIGH RISK ({prob:.2f}%)")
    else:
        st.success(f"✅ LOW RISK ({prob:.2f}%)")