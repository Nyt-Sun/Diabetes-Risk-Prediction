import streamlit as st
import numpy as np
import pickle
import os

st.title("🩺 Diabetes Risk Prediction App")

st.sidebar.title("About App")
st.sidebar.info("""
This app predicts diabetes risk using machine learning trained on the Pima Indians dataset.
""")

# Check if model exists
model_path = "model/model.pkl"
scaler_path = "model/scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("Model files not found. Run train.py first.")
    st.stop()

# Load model safely
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

st.write("Enter patient details below:")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 100, 30)

if st.button("Predict"):
    data = np.array([[pregnancies, glucose, blood_pressure,
                      skin_thickness, insulin, bmi, dpf, age]])

    data = scaler.transform(data)

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Diabetes")
    else:
        st.success(f"✅ Low Risk of Diabetes")

    st.info(f"""
    Confidence Scores:
    - No Diabetes: {probability[0]*100:.2f}%
    - Diabetes: {probability[1]*100:.2f}%
    """)

    st.subheader("How it works")
st.write("""
The model uses Logistic Regression trained on medical features such as glucose, BMI, and insulin levels.
It predicts the probability of diabetes risk.
""")