import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL & SCALER
# =========================

model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("🩺 Diabetes Risk Prediction System")
st.markdown("Machine Learning-based diabetes risk prediction using the Pima Indians dataset.")

# =========================
# TABS
# =========================

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Insights", "📂 Dataset"])

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("Patient Inputs")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 200, 120)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 30)

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

input_scaled = scaler.transform(input_data)

# =========================
# TAB 1 - PREDICTION
# =========================

with tab1:

    st.subheader("Prediction Result")

    if st.button("Predict Diabetes Risk"):

        prediction = model.predict(input_scaled)

        # Safe probability handling
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = None

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

        if probability is not None:
            st.metric("Risk Probability", f"{probability:.2%}")
            st.progress(float(probability))

# =========================
# TAB 2 - MODEL INSIGHTS
# =========================

with tab2:

    st.subheader("Model Insights")

    # Feature importance (SAFE CHECK)
    if hasattr(model, "feature_importances_"):

        features = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.set_title("Feature Importance")

        st.pyplot(fig)

    else:
        st.info("Feature importance is not available for this model.")

# =========================
# TAB 3 - DATASET
# =========================

with tab3:

    st.subheader("Dataset Preview")

    df = pd.read_csv("data/diabetes.csv")

    st.dataframe(df.head())

    st.write("Shape:", df.shape)

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption("AI/ML Capstone Project • Diabetes Risk Prediction")