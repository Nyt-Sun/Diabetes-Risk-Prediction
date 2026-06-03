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
page_icon="⚕️",
layout="wide"
)

# =====================================

# CUSTOM CSS

# =====================================

st.markdown("""

<style>

/* APP BACKGROUND */
.stApp {
    background-color: #f5f0e6;
}

/* HEADER */
.header {
    background-color: #0b1f3a;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
}

.logo {
    font-size: 70px;
    margin-bottom: 5px;
}

.title {
    color: white;
    font-size: 60px;
    font-weight: 900;
    line-height: 1.1;
}

.subtitle {
    color: #d6d6d6;
    font-size: 15px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #e3d7c3 !important;
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: #1f1f1f;
}

/* INPUTS */
[data-testid="stNumberInput"] {
    width: 100%;
    margin-bottom: 8px;
}

.stNumberInput input {
    text-align: center;
}

/* BUTTON */
.stButton > button {
    background-color: #0b1f3a;
    color: white;
    font-size: 18px;
    font-weight: 800;
    border-radius: 12px;
    border: none;
    height: 55px;
}

.stButton > button:hover {
    background-color: #163a63;
}

/* MAIN AREA */
.block-container {
    background-color: #f8f3ea;
    border-radius: 10px;
    padding: 2rem;
}

</style>

""", unsafe_allow_html=True)

# =====================================

# HEADER

# =====================================

st.markdown("""

<div class="header">
    <div class="logo">⚕️</div>
    <div class="title">Diabetes Risk Predictor</div>
    <div class="subtitle">
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

st.sidebar.markdown("## 🧾 Patient Clinical Data")

left, right = st.sidebar.columns(2)

with left:
pregnancies = st.number_input(
"Pregnancies", 0, 20, 1
)

```
blood_pressure = st.number_input(
    "Blood Pressure", 0, 150, 70
)

insulin = st.number_input(
    "Insulin", 0, 900, 80
)

age = st.number_input(
    "Age", 1, 120, 30
)
```

with right:
glucose = st.number_input(
"Glucose", 0, 300, 120
)

```
skin_thickness = st.number_input(
    "Skin Thickness", 0, 100, 20
)

bmi = st.number_input(
    "BMI", 0.0, 70.0, 25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    0.0,
    3.0,
    0.50
)
```

st.sidebar.markdown("<br>", unsafe_allow_html=True)

b1, b2, b3 = st.sidebar.columns([1,3,1])

with b2:
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
"📄 Clinical Report"
])

# =====================================

# PREDICTION

# =====================================

if predict:

```
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

prediction = model.predict(
    scaled_features
)[0]

probability = model.predict_proba(
    scaled_features
)[0][1] * 100
```

# =====================================

# TAB 1

# =====================================

with tab1:

```
st.subheader("Patient Risk Assessment")

if predict:

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

    st.subheader(
        "🧠 Key Medical Indicators"
    )

    chart_df = pd.DataFrame({
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
        chart_df.set_index("Feature")
    )

else:
    st.info(
        "Enter patient data and click RUN ANALYSIS."
    )
```

# =====================================

# TAB 2

# =====================================

with tab2:

```
st.subheader(
    "📈 Model Evaluation Dashboard"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", "77%")
m2.metric("Precision", "75%")
m3.metric("Recall", "73%")
m4.metric("F1 Score", "74%")

metrics = {
    "Accuracy": 0.77,
    "Precision": 0.75,
    "Recall": 0.73,
    "F1 Score": 0.74
}

fig, ax = plt.subplots(
    figsize=(8,4)
)

ax.bar(
    metrics.keys(),
    metrics.values(),
    color="#0b1f3a"
)

ax.set_ylim(0,1)
ax.set_title(
    "Model Performance Metrics"
)

st.pyplot(fig)
```

# =====================================

# TAB 3

# =====================================

with tab3:

```
st.subheader(
    "Clinical Report Generator"
)

if predict:

    status = (
        "HIGH RISK"
        if prediction == 1
        else "LOW RISK"
    )

    report = f"""
```

# DIABETES RISK CLINICAL REPORT

Pregnancies: {pregnancies}
Glucose: {glucose}
Blood Pressure: {blood_pressure}
Skin Thickness: {skin_thickness}
Insulin: {insulin}
BMI: {bmi}
Diabetes Pedigree Function: {dpf}
Age: {age}

Risk Probability: {probability:.2f}%

FINAL STATUS: {status}
"""

```
    st.download_button(
        "📄 Download Clinical Report",
        report,
        file_name="diabetes_clinical_report.txt",
        mime="text/plain"
    )

else:
    st.info(
        "Run analysis to generate a report."
    )
```
