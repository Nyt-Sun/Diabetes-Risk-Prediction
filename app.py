import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide",
    page_icon="⚕️"
)

# =============================
# CUSTOM STYLE
# =============================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: #f5f0e6;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #e3d7c3 !important;
    min-width: 430px !important;
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: #1f1f1f;
}

/* HEADER */
.header {
    background-color: #0b1f3a;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
}

.title {
    color: white;
    font-size: 60px;
    font-weight: 900;
}

.subtitle {
    color: #d6d6d6;
    font-size: 14px;
}

/* MAIN CONTENT */
.block-container {
    background-color: #ffffff;
    padding: 2rem;
}

/* =============================
   DEFAULT BUTTONS (NOT AFFECT RUN ANALYSIS)
============================= */
div.stButton > button {
    background-color: #0b1f3a;
    color: white;
    font-weight: 600;
    border-radius: 10px;
}

/* =============================
   ONLY RUN ANALYSIS BUTTON (SIDEBAR ONLY)
============================= */
section[data-testid="stSidebar"] div.stButton > button {
    background-color: #d90429 !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    height: 60px !important;
    width: 100% !important;
    border-radius: 12px !important;
}

/* HOVER EFFECT FOR RUN ANALYSIS */
section[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #a1031f !important;
}

</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown("""
<div class="header">
    <div class="title">⚕️ Diabetes Risk Predictor</div>
    <div class="subtitle">AI Clinical Decision Support System</div>
</div>
""", unsafe_allow_html=True)

st.write("---")

# =============================
# LOAD MODEL
# =============================
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.markdown("### 🧾 Patient Clinical Data")

col1, col2 = st.sidebar.columns(2)

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
# RUN ANALYSIS (ONLY THIS BUTTON IS RED)
# =============================
predict = st.sidebar.button("🔍 RUN ANALYSIS")

# =============================
# MODEL LOGIC
# =============================
if predict:

    features = np.array([[
        pregnancies, glucose, bp, skin,
        insulin, bmi, dpf, age
    ]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1] * 100

# =============================
# TABS
# =============================
tab1, tab2, tab3 = st.tabs([
    "📊 Patient Diagnosis",
    "📈 Model Performance",
    "🧾 Clinical Report"
])

# =============================
# TAB 1
# =============================
with tab1:

    st.subheader("Patient Risk Assessment")

    if predict:

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Risk Probability", f"{prob:.2f}%")

        with c2:
            if prediction == 1:
                st.error("🚨 HIGH RISK")
            else:
                st.success("✅ LOW RISK")

        st.divider()

        st.subheader("🧠 Key Medical Indicators")

        df = pd.DataFrame({
            "Feature": [
                "Pregnancies","Glucose","Blood Pressure",
                "Skin Thickness","Insulin","BMI","DPF","Age"
            ],
            "Value": [
                pregnancies, glucose, bp, skin,
                insulin, bmi, dpf, age
            ]
        })

        st.bar_chart(df.set_index("Feature"))

    else:
        st.info("Enter patient data and click RUN ANALYSIS.")

# =============================
# TAB 2
# =============================
with tab2:

    st.subheader("📈 Model Evaluation Dashboard")

    metrics = {
        "Accuracy": 0.77,
        "Precision": 0.75,
        "Recall": 0.73,
        "F1 Score": 0.74
    }

    colA, colB = st.columns(2)

    with colA:
        st.metric("Accuracy", "77%")
        st.metric("Precision", "75%")
        st.metric("Recall", "73%")
        st.metric("F1 Score", "74%")

    with colB:
        fig, ax = plt.subplots()
        ax.bar(metrics.keys(), metrics.values(), color="#0b1f3a")
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance")
        st.pyplot(fig)

# =============================
# TAB 3
# =============================
with tab3:

    st.subheader("🧾 Clinical Report")

    if predict:

        status = "HIGH RISK" if prediction == 1 else "LOW RISK"

        report = f"""
DIABETES RISK CLINICAL REPORT
================================

Pregnancies: {pregnancies}
Glucose: {glucose}
Blood Pressure: {bp}
Skin Thickness: {skin}
Insulin: {insulin}
BMI: {bmi}
Diabetes Pedigree Function: {dpf}
Age: {age}

FINAL STATUS: {status}
"""

        st.download_button(
            "📄 Download Clinical Report",
            report,
            file_name="diabetes_report.txt"
        )

    else:
        st.info("Run analysis to generate clinical report.")