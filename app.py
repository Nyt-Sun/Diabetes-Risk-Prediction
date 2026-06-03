import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model/diabetes_model.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction System")
st.write("Enter patient details below to predict diabetes risk.")

# Sidebar info
st.sidebar.header("About the App")
st.sidebar.write(
    "This app uses a Machine Learning model trained on the Pima Indians Diabetes Dataset."
)

# Inputs
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Prediction button
if st.button("Predict"):

    features = np.array([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    # Result
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Diabetes\nProbability: {probability[0][1]*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Diabetes\nProbability: {probability[0][0]*100:.2f}%")

    # Store result for download
    result_df = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DPF": [dpf],
        "Age": [age],
        "Prediction": [int(prediction[0])]
    })

    st.download_button(
        "⬇ Download Result as CSV",
        result_df.to_csv(index=False),
        "diabetes_prediction.csv",
        "text/csv"
    )