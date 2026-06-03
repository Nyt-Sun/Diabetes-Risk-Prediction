import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# BACKGROUND STYLE (clinical soft theme)
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #f5f7fa, #e4edf5);
    }

    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #0b3d91;
        text-align: center;
        padding: 10px;
    }

    .sub-title {
        text-align: center;
        color: #555;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<div class="main-title">Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Clinical AI Decision Support System</div>', unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# -----------------------------
# SIDEBAR (PATIENT INPUT)
# -----------------------------
st.sidebar.header("🧾 Patient Clinical Data")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 200, 120)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.number_input("Age", 1, 100, 30)

predict_btn = st.sidebar.button("🔍 Predict Risk")

# -----------------------------
# MAIN PANEL
# -----------------------------
st.write("## 🏥 Clinical Assessment Dashboard")

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

    # -----------------------------
    # RESULT CARD
    # -----------------------------
    st.write("### 📊 Patient Risk Report")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Risk Probability", f"{prob:.2f}%")

    with col2:
        if prediction[0] == 1:
            st.error("🚨 HIGH RISK")
        else:
            st.success("✅ LOW RISK")

    st.write("---")

    # -----------------------------
    # CLINICAL SUMMARY
    # -----------------------------
    if prediction[0] == 1:
        st.markdown(
            """
            ### 🧠 Clinical Interpretation
            - Elevated diabetes risk detected
            - Recommend immediate clinical follow-up
            - Lifestyle and metabolic review required
            """
        )
    else:
        st.markdown(
            """
            ### 🧠 Clinical Interpretation
            - Low probability of diabetes
            - No immediate clinical concern
            - Routine monitoring advised
            """
        )

    # -----------------------------
    # FEATURE SNAPSHOT
    # -----------------------------
    st.write("### 📈 Patient Indicator Snapshot")

    df = pd.DataFrame({
        "Feature": [
            "Pregnancies", "Glucose", "Blood Pressure",
            "Skin Thickness", "Insulin", "BMI",
            "DPF", "Age"
        ],
        "Value": [
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi,
            dpf, age
        ]
    })

    st.bar_chart(df.set_index("Feature"))

else:
    st.info("Enter patient data in the sidebar and click **Predict Risk** to generate report.")