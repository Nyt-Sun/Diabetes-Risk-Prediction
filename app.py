import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #f5f0e6;
}

/* HEADER */
.header-container {
    background-color: #0b1f3a;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
}

.header-title {
    color: white;
    font-size: 48px;
    font-weight: 800;
}

.header-subtitle {
    color: #d8d8d8;
    font-size: 16px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #e3d7c3;
}

/* SIDEBAR HEADER */
.sidebar-title {
    text-align: center;
    color: #0b1f3a;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* INPUT BOXES */
[data-testid="stNumberInput"] {
    margin-bottom: 8px;
}

.stNumberInput input {
    text-align: center;
}

/* BUTTON */
div.stButton > button {
    width: 100%;
    background-color: #0b1f3a;
    color: white;
    font-weight: 800;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px;
    border: none;
}

div.stButton > button:hover {
    background-color: #163a63;
}

/* MAIN CONTENT */
.block-container {
    background-color: #f8f3ea;
    border-radius: 12px;
    padding: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown("""
<div class="header-container">
    <div class="header-title">🏥 Diabetes Risk Predictor</div>
    <div class="header-subtitle">
        AI Clinical Decision Support System
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# =====================================
# SIDEBAR
# =====================================
st.sidebar.markdown(
    '<div class="sidebar-title">🧾 Patient Clinical Data</div>',
    unsafe_allow_html=True
)

pregnancies = st.sidebar.number_input(
    "Pregnancies", 0, 20, 1
)

glucose = st.sidebar.number_input(
    "Glucose", 0, 300, 120
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure", 0, 200, 70
)

skin_thickness = st.sidebar.number_input(
    "Skin Thickness", 0, 100, 20
)

insulin = st.sidebar.number_input(
    "Insulin", 0, 900, 80
)

bmi = st.sidebar.number_input(
    "BMI", 0.0, 70.0, 25.0
)

dpf = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    0.0,
    3.0,
    0.50
)

age = st.sidebar.number_input(
    "Age", 1, 120, 30
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# CENTERED BUTTON
left, center, right = st.sidebar.columns([1, 4, 1])

with center:
    predict = st.button(
        "🔍 RUN ANALYSIS",
        use_container_width=True
    )

# =====================================
# TABS
# =====================================
tab1, tab2, tab3 = st.tabs([
    "📊 Patient Diagnosis",
    "📈 Model Performance",
    "🧾 Clinical Report"
])

# =====================================
# PATIENT DIAGNOSIS
# =====================================
with tab1:

    st.subheader("Patient Risk Assessment")

    if predict:

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

        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1] * 100

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Risk Probability",
                f"{probability:.2f}%"
            )

        with c2:
            if prediction == 1:
                st.error("🚨 HIGH RISK")
            else:
                st.success("✅ LOW RISK")

        st.divider()

        st.subheader("🧠 Key Medical Indicators")

        indicators = pd.DataFrame({
            "Feature": [
                "Pregnancies",
                "Glucose",
                "Blood Pressure",
                "Skin Thickness",
                "Insulin",
                "BMI",
                "DPF",
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

        st.bar_chart(
            indicators.set_index("Feature")
        )

    else:
        st.info(
            "Enter patient information and click RUN ANALYSIS."
        )

# =====================================
# MODEL PERFORMANCE
# =====================================
with tab2:

    st.subheader("📈 Model Evaluation Dashboard")

    metrics = {
        "Accuracy": 0.77,
        "Precision": 0.75,
        "Recall": 0.73,
        "F1 Score": 0.74
    }

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Accuracy",
        f"{metrics['Accuracy']*100:.1f}%"
    )

    m2.metric(
        "Precision",
        f"{metrics['Precision']*100:.1f}%"
    )

    m3.metric(
        "Recall",
        f"{metrics['Recall']*100:.1f}%"
    )

    m4.metric(
        "F1 Score",
        f"{metrics['F1 Score']*100:.1f}%"
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        metrics.keys(),
        metrics.values(),
        color="#0b1f3a"
    )

    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Metrics")

    st.pyplot(fig)

# =====================================
# CLINICAL REPORT
# =====================================
with tab3:

    st.subheader("🧾 Clinical Report")

    if predict:

        report = f"""
DIABETES RISK CLINICAL REPORT
================================

Pregnancies: {pregnancies}
Glucose: {glucose}
Blood Pressure: {blood_pressure}
Skin Thickness: {skin_thickness}
Insulin: {insulin}
BMI: {bmi}
Diabetes Pedigree Function: {dpf}
Age: {age}

Risk Probability: {probability:.2f}%

Risk Status:
{"HIGH RISK" if prediction == 1 else "LOW RISK"}

Generated by:
Diabetes Risk Predictor
"""

        st.download_button(
            label="📄 Download Clinical Report",
            data=report,
            file_name="clinical_report.txt",
            mime="text/plain"
        )

    else:
        st.info(
            "Run analysis first to generate a report."
        )