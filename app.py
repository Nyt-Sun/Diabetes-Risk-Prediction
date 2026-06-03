import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load model and scaler
# -----------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# -----------------------------
# App Title
# -----------------------------
st.title("🩺 Diabetes Risk Prediction System")
st.write("Enter patient details below to predict diabetes risk.")

# -----------------------------
# Input Fields
# -----------------------------
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 100, 30)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    # Input array
    features = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Scale features (VERY IMPORTANT)
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠ High Risk of Diabetes")
        st.write(f"Confidence: {probability[0][1] * 100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Diabetes")
        st.write(f"Confidence: {probability[0][0] * 100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with Streamlit | Diabetes Risk Prediction ML App")