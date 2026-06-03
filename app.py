import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Hospital AI - Diabetes Risk System",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# STYLE (Hospital UI Theme)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #eef2f7, #dbe9f4);
}

.block-container {
    padding-top: 2rem;
}

h1 {
    color: #0b3d91;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# -----------------------------
# HEADER
# -----------------------------
st.title("🏥 Hospital AI Diabetes Risk System")
st.caption("Clinical Decision Support | AI-powered risk stratification engine")

st.write("---")

# -----------------------------
# SIDEBAR INPUT (CLINICAL INTAKE)
# -----------------------------
st.sidebar.header("🧾 Patient Intake Form")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 200, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.number_input("Age", 1, 100, 30)

predict = st.sidebar.button("🔍 Run Clinical Analysis")

# -----------------------------
# MAIN LAYOUT
# -----------------------------
col1, col2 = st.columns([1.2, 1])

if predict:

    # -------------------------
    # PREDICTION
    # -------------------------
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    risk_percent = prob * 100

    # -------------------------
    # LEFT PANEL (RESULTS)
    # -------------------------
    with col1:

        st.subheader("📊 Patient Risk Dashboard")

        st.metric("🧪 Diabetes Risk Score", f"{risk_percent:.2f}%")

        # Risk classification
        if pred == 1:
            st.error("🚨 HIGH RISK PATIENT")
            status = "HIGH RISK"
        else:
            st.success("✅ LOW RISK PATIENT")
            status = "LOW RISK"

        # -------------------------
        # GAUGE STYLE (SIMULATED)
        # -------------------------
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.barh(["Risk"], [risk_percent], color="red" if pred == 1 else "green")
        ax.set_xlim(0, 100)
        ax.set_title("Risk Level Gauge")
        st.pyplot(fig)

        # -------------------------
        # CLINICAL INTERPRETATION
        # -------------------------
        st.subheader("🧠 Clinical Interpretation")

        if pred == 1:
            st.warning("""
            - Elevated glucose and metabolic markers detected  
            - Patient shows diabetes risk indicators  
            - Recommend clinical confirmation tests (HbA1c, fasting glucose)  
            """)
        else:
            st.info("""
            - No strong diabetic indicators detected  
            - Routine monitoring recommended  
            - Maintain healthy lifestyle and periodic screening  
            """)

    # -------------------------
    # RIGHT PANEL (ADVANCED INSIGHTS)
    # -------------------------
    with col2:

        st.subheader("📈 Model Confidence")

        st.write(f"Probability (Diabetic): **{prob*100:.2f}%**")
        st.write(f"Probability (Non-Diabetic): **{(1-prob)*100:.2f}%**")

        st.progress(float(prob))

        st.write("---")

        # -------------------------
        # FEATURE SNAPSHOT
        # -------------------------
        st.subheader("🧬 Patient Biomarker Profile")

        features_df = pd.DataFrame({
            "Feature": [
                "Pregnancies", "Glucose", "Blood Pressure",
                "Skin Thickness", "Insulin", "BMI",
                "DPF", "Age"
            ],
            "Value": [
                pregnancies, glucose, bp, skin,
                insulin, bmi, dpf, age
            ]
        })

        st.bar_chart(features_df.set_index("Feature"))

        # -------------------------
        # REPORT DOWNLOAD
        # -------------------------
        report = f"""
        HOSPITAL AI DIABETES REPORT

        Risk Status: {status}
        Risk Score: {risk_percent:.2f}%

        Patient Data:
        - Pregnancies: {pregnancies}
        - Glucose: {glucose}
        - Blood Pressure: {bp}
        - BMI: {bmi}
        - Age: {age}
        """

        st.download_button(
            "📄 Download Patient Report",
            report,
            file_name="diabetes_report.txt"
        )

else:
    st.info("Enter patient details in the sidebar and click **Run Clinical Analysis** to generate hospital report.")