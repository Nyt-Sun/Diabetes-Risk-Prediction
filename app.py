import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide",
    page_icon="🏥"
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #eef3f8, #d9e6f2);
}

/* HEADER */
.header {
    background-color: #0b1f3a;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
}

.title {
    color: white;
    font-size: 46px;
    font-weight: 800;
}

.subtitle {
    color: #cfd8e3;
    font-size: 14px;
}

/* COMPACT SIDEBAR */
section[data-testid="stSidebar"] {
    width: 320px !important;
}

.small-space {
    margin-bottom: -10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <div class="title">🏥 Diabetes Risk Predictor</div>
    <div class="subtitle">AI Clinical Decision Support System</div>
</div>
""", unsafe_allow_html=True)

st.write("---")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# -----------------------------
# COMPACT SIDEBAR GRID INPUT
# -----------------------------
st.sidebar.header("🧾 Patient Clinical Data")

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
    dpf = st.number_input("DPF", 0.0, 2.5, 0.5)

predict = st.sidebar.button("🔍 Run Analysis")

# -----------------------------
# TABS
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

    if predict:

        features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled = scaler.transform(features)

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100

        colA, colB = st.columns(2)

        with colA:
            st.metric("Risk Probability", f"{prob:.2f}%")

        with colB:
            if pred == 1:
                st.error("🚨 HIGH RISK")
            else:
                st.success("✅ LOW RISK")

        st.write("---")

        st.subheader("🧠 Key Medical Indicators")

        df = pd.DataFrame({
            "Feature": ["Pregnancies","Glucose","BP","Skin","Insulin","BMI","DPF","Age"],
            "Value": [pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]
        })

        st.bar_chart(df.set_index("Feature"))

    else:
        st.info("All patient fields are visible. Click **Run Analysis** to generate diagnosis.")

# =========================================================
# TAB 2 — MODEL PERFORMANCE (GRAPHICAL)
# =========================================================
with tab2:

    st.subheader("📈 Model Evaluation Dashboard")

    metrics = {
        "Accuracy": 0.77,
        "Precision": 0.75,
        "Recall": 0.73,
        "F1 Score": 0.74
    }

    col1, col2 = st.columns(2)

    with col1:
        for k, v in metrics.items():
            st.metric(k, f"{v*100:.2f}%")

    with col2:
        fig, ax = plt.subplots()
        ax.bar(metrics.keys(), metrics.values(), color="#0b1f3a")
        ax.set_ylim(0, 1)
        ax.set_title("Performance Chart")
        plt.xticks(rotation=25)
        st.pyplot(fig)

# =========================================================
# TAB 3 — CLINICAL REPORT
# =========================================================
with tab3:

    st.subheader("🧾 Clinical Report")

    if predict:

        report = f"""
DIABETES RISK REPORT
----------------------
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
            "📄 Download Report",
            report,
            file_name="diabetes_report.txt"
        )

    else:
        st.info("Run analysis first to generate clinical report.")