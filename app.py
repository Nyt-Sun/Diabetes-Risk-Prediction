import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from reportlab.pdfgen import canvas

# =========================
# PAGE CONFIG (HOSPITAL STYLE)
# =========================

st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL
# =========================

model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
X_test = pickle.load(open("model/X_test.pkl", "rb"))
y_test = pickle.load(open("model/y_test.pkl", "rb"))

# =========================
# HEADER (HOSPITAL STYLE)
# =========================

st.markdown("""
    <div style="background-color:#0f4c81;padding:15px;border-radius:10px">
        <h2 style="color:white;text-align:center;">
            🏥 AI Clinical Decision Support System
        </h2>
    </div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# SESSION STATE
# =========================

if "pred" not in st.session_state:
    st.session_state.pred = None

if "prob" not in st.session_state:
    st.session_state.prob = None

# =========================
# SIDEBAR (PATIENT INPUT PANEL)
# =========================

st.sidebar.header("🧾 Patient Clinical Data")

labels = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

inputs = []

for label in labels:
    value = st.sidebar.number_input(label, value=1.0)
    inputs.append(value)

input_array = np.array([inputs])
input_scaled = scaler.transform(input_array)

predict_btn = st.sidebar.button("🔍 Run Diagnosis")

# =========================
# MAIN DASHBOARD TABS
# =========================

tab1, tab2, tab3 = st.tabs([
    "📊 Patient Diagnosis",
    "📈 Model Performance",
    "🧠 Clinical Report"
])

# =========================
# TAB 1 - DIAGNOSIS DASHBOARD
# =========================

with tab1:

    st.subheader("Patient Risk Assessment")

    if predict_btn:

        st.session_state.pred = model.predict(input_scaled)[0]
        st.session_state.prob = model.predict_proba(input_scaled)[0][1]

    if st.session_state.prob is not None:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Risk Probability", f"{st.session_state.prob:.2%}")

        with col2:
            risk_level = "HIGH RISK" if st.session_state.pred == 1 else "LOW RISK"
            st.metric("Risk Status", risk_level)

        with col3:
            st.metric("Clinical Confidence", "AI Model Output")

        # =========================
        # COLORED ALERT BOX
        # =========================

        if st.session_state.pred == 1:
            st.error("⚠️ HIGH RISK DETECTED — Immediate clinical attention recommended.")
        else:
            st.success("✅ LOW RISK — No immediate concern detected.")

        st.divider()

        # =========================
        # FEATURE IMPORTANCE
        # =========================

        if hasattr(model, "feature_importances_"):

            st.subheader("🧠 Key Medical Indicators")

            importance = pd.DataFrame({
                "Feature": labels,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance.set_index("Feature"))

# =========================
# TAB 2 - MODEL PERFORMANCE
# =========================

with tab2:

    st.subheader("Model Evaluation Dashboard")

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("ROC Curve")

        y_score = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_score = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.legend()
        ax.set_title("ROC Curve")

        st.pyplot(fig)

# =========================
# TAB 3 - CLINICAL REPORT
# =========================

with tab3:

    st.subheader("Medical Report Generator")

    st.info("Generate a downloadable clinical summary for patient record.")

    if st.button("Generate Report PDF"):

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer)

        c.drawString(100, 800, "AI Clinical Diabetes Risk Report")

        pred_text = (
            "HIGH RISK" if st.session_state.pred == 1 else "LOW RISK"
        ) if st.session_state.pred is not None else "Not Generated"

        prob_text = (
            f"{st.session_state.prob:.2%}"
        ) if st.session_state.prob is not None else "N/A"

        c.drawString(100, 780, f"Diagnosis: {pred_text}")
        c.drawString(100, 760, f"Probability: {prob_text}")

        c.drawString(100, 720, "Recommendation: Consult healthcare provider if high risk.")

        c.save()
        buffer.seek(0)

        st.download_button(
            "Download Clinical Report",
            buffer,
            file_name="clinical_report.pdf",
            mime="application/pdf"
        )