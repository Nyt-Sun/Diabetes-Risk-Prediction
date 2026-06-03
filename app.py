import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Clinical Dashboard",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Load Model + Scaler
# -----------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# -----------------------------
# Header (Clinical Style)
# -----------------------------
st.markdown(
    """
    <div style="background-color:#0b3d91;padding:20px;border-radius:10px">
        <h1 style="color:white;text-align:center;">
            🏥 Diabetes Risk Clinical Decision Support System
        </h1>
        <p style="color:white;text-align:center;">
            AI-powered patient risk assessment tool for clinical screening
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# -----------------------------
# Layout Columns (Clinical Form Style)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Clinical Data")

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Plasma Glucose Level (mg/dL)", 0, 200, 120)
    blood_pressure = st.number_input("Diastolic Blood Pressure (mmHg)", 0, 150, 70)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)

with col2:
    st.subheader("🧪 Metabolic Indicators")

    insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("Body Mass Index (BMI)", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age (years)", 1, 100, 30)

st.write("---")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Run Clinical Risk Assessment"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    risk_score = prob[0][1] * 100

    # -----------------------------
    # Clinical Result Panel
    # -----------------------------
    st.subheader("🧾 Clinical Assessment Result")

    if prediction[0] == 1:

        st.markdown(
            f"""
            <div style="background-color:#ffcccc;padding:20px;border-radius:10px">
                <h2 style="color:#b30000;">⚠ High Diabetes Risk Detected</h2>
                <h3>Risk Probability: {risk_score:.2f}%</h3>
                <p>Recommendation: Immediate clinical follow-up advised.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div style="background-color:#ccffcc;padding:20px;border-radius:10px">
                <h2 style="color:#006600;">✅ Low Diabetes Risk</h2>
                <h3>Risk Probability: {100 - risk_score:.2f}%</h3>
                <p>Recommendation: Routine monitoring suggested.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Footer (Clinical Note)
# -----------------------------
st.write("---")
st.caption("Clinical Decision Support System | AI/ML Capstone Project | Not for real medical diagnosis")