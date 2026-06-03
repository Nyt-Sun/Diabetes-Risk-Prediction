import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Clinical Decision Support System",
    layout="wide",
    page_icon="🏥"
)

# -----------------------------
# Load Model + Scaler
# -----------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# -----------------------------
# HEADER
# -----------------------------
st.title("🏥 AI Clinical Decision Support System")
st.write("AI-powered diabetes risk screening and clinical decision support tool.")

st.write("---")

# -----------------------------
# LAYOUT
# -----------------------------
col1, col2 = st.columns([1, 1])

# -----------------------------
# PATIENT INPUT SECTION
# -----------------------------
with col1:
    st.subheader("🧾 Patient Clinical Data")

    pregnancies = st.number_input("Pregnancies", 0.0, 20.0, 1.0)
    glucose = st.number_input("Glucose", 0.0, 200.0, 120.0)
    blood_pressure = st.number_input("BloodPressure", 0.0, 150.0, 70.0)
    skin_thickness = st.number_input("SkinThickness", 0.0, 100.0, 20.0)
    insulin = st.number_input("Insulin", 0.0, 900.0, 80.0)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("DiabetesPedigreeFunction", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1.0, 100.0, 30.0)

    predict_btn = st.button("🔍 Generate Clinical Report")

# -----------------------------
# RIGHT PANEL (RESULTS)
# -----------------------------
with col2:
    st.subheader("📊 Model Performance")
    st.info("Random Forest Classifier trained on Pima Indians Diabetes Dataset")
    st.write("Accuracy: ~77% (example placeholder)")

    st.subheader("🧠 Clinical Report")

    if predict_btn:

        # input array
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

        # scale
        features_scaled = scaler.transform(features)

        # prediction
        prediction = model.predict(features_scaled)
        prob = model.predict_proba(features_scaled)[0][1] * 100

        st.write("### Patient Risk Assessment")

        st.metric(label="Risk Probability", value=f"{prob:.2f}%")

        # -----------------------------
        # Risk Status
        # -----------------------------
        if prediction[0] == 1:
            st.error("🚨 HIGH RISK")
            st.markdown("**Clinical Confidence:** High probability of diabetes detected.")
        else:
            st.success("✅ LOW RISK")
            st.markdown("**Clinical Confidence:** No immediate concern detected.")

        st.write("---")

        # -----------------------------
        # SIMPLE FEATURE IMPORTANCE VISUAL (STATIC STYLE)
        # -----------------------------
        st.write("🧠 Key Medical Indicators (Input Snapshot)")

        features_df = pd.DataFrame({
            "Feature": [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigree",
                "Age"
            ],
            "Value": [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age
            ]
        })

        st.bar_chart(features_df.set_index("Feature"))

    else:
        st.info("Enter patient data and click **Generate Clinical Report**")