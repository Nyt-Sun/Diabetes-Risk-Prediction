import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide",
    page_icon="🏥"
)

# -----------------------------
# CUSTOM HEADER (NAVY + WHITE TITLE)
# -----------------------------
st.markdown("""
<style>
.header {
    background-color: #0b1f3a;
    padding: 25px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.title {
    color: white;
    font-size: 36px;
    font-weight: bold;
}

.sub {
    color: #cfd8e3;
    font-size: 14px;
}

img {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

colA, colB = st.columns([2, 1])

with colA:
    st.markdown("""
    <div class="header">
        <div>
            <div class="title">Diabetes Risk Predictor</div>
            <div class="sub">AI-powered Clinical Decision Support System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with colB:
    st.image("https://images.unsplash.com/photo-1582719478250-c89cae4dc85b",
             use_container_width=True)

st.write("---")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# -----------------------------
# SIDEBAR INPUT (CLEAN)
# -----------------------------
st.sidebar.header("🧾 Patient Data Input")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 200, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.number_input("Age", 1, 100, 30)

predict = st.sidebar.button("🔍 Run Analysis")

# -----------------------------
# TABS (FOLDERS)
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Patient Diagnosis",
    "📈 Model Performance",
    "🧾 Clinical Report"
])

# =========================================================
# TAB 1 — PATIENT DIAGNOSIS
# =========================================================
with tab1:

    st.subheader("Patient Risk Assessment")

    if predict:

        features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled = scaler.transform(features)

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Risk Probability", f"{prob:.2f}%")

        with col2:
            if pred == 1:
                st.error("🚨 HIGH RISK")
            else:
                st.success("✅ LOW RISK")

        st.write("---")

        st.subheader("🧠 Key Medical Indicators")

        df = pd.DataFrame({
            "Feature": [
                "Pregnancies", "Glucose", "Blood Pressure",
                "Skin Thickness", "Insulin", "BMI", "DPF", "Age"
            ],
            "Value": [
                pregnancies, glucose, bp, skin, insulin, bmi, dpf, age
            ]
        })

        st.bar_chart(df.set_index("Feature"))

    else:
        st.info("Enter patient data and click **Run Analysis**.")

# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================
with tab2:

    st.subheader("Model Evaluation Dashboard")

    st.write("Random Forest Classifier trained on Pima Indians Diabetes Dataset")

    st.metric("Accuracy", "≈ 77%")
    st.metric("Model Type", "Random Forest")
    st.metric("Dataset", "Pima Indians Diabetes")

    st.write("---")

    st.subheader("Performance Insight")

    st.write("""
    - Model performs well on balanced classification tasks  
    - Higher sensitivity to glucose levels and BMI  
    - Suitable for binary medical risk prediction  
    """)

# =========================================================
# TAB 3 — CLINICAL REPORT
# =========================================================
with tab3:

    st.subheader("Clinical Report Generator")

    if predict:

        report = f"""
DIABETES RISK CLINICAL REPORT
--------------------------------
Pregnancies: {pregnancies}
Glucose: {glucose}
Blood Pressure: {bp}
Skin Thickness: {skin}
Insulin: {insulin}
BMI: {bmi}
DPF: {dpf}
Age: {age}

STATUS: {"HIGH RISK" if model.predict(scaler.transform([[pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]]))[0] == 1 else "LOW RISK"}
        """

        st.download_button(
            "📄 Download Clinical Report",
            report,
            file_name="diabetes_clinical_report.txt"
        )

    else:
        st.info("Run analysis first to generate report.")