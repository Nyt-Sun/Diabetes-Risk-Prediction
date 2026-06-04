🏥 Diabetes Risk Predictor (Clinical AI Dashboard)

A machine learning–powered clinical decision support system that predicts the likelihood of diabetes using patient health indicators. Built with Python, Scikit-learn, and Streamlit, the system simulates a hospital-style EMR dashboard for real-time risk assessment.

📌 Project Overview

This project uses the Pima Indians Diabetes Dataset to train a machine learning model that predicts whether a patient is at risk of diabetes based on diagnostic health features.

The system is deployed as an interactive Streamlit web application designed with a clinical hospital interface.

🎯 Objective

To build an AI-assisted medical tool that:

- Predicts diabetes risk in real time  
- Visualizes patient health indicators  
- Provides a clinical-style report  
- Demonstrates end-to-end ML deployment  

🧠 Machine Learning Model

- Algorithm: Random Forest Classifier  
- Framework: Scikit-learn  
- Preprocessing:
  - Invalid zero values in medical features were treated as missing values (NaN)
  - Missing values were filled using **median imputation (training data only)**
  - Feature scaling using StandardScaler
- Output: Binary classification (Low Risk / High Risk)

📊 Dataset

- Name: Pima Indians Diabetes Dataset  
- Source: National Institute of Diabetes and Digestive and Kidney Diseases  

Features:
- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

🖥️ Application Features

🧾 Patient Clinical Input Panel  
- Compact sidebar form for entering patient data  
- Structured medical input fields  

🔍 Risk Prediction Engine  
- Real-time diabetes risk prediction  
- Probability score output  
- Classification: LOW RISK / HIGH RISK  

📊 Patient Diagnosis Dashboard  
- Risk probability display  
- Key medical indicators visualization (bar chart)  

📈 Model Performance Dashboard  
- Accuracy, Precision, Recall, F1-score  
- Graphical performance visualization  

🧾 Clinical Report Generator  
- Downloadable patient report (TXT format)  
- Includes all input values + prediction result  

🎨 UI / UX Design  
- Clinical hospital-style interface  
- Beige-themed background with darker sidebar  
- Centered and compact patient input panel  
- Red “Run Analysis” action button  
- EMR-inspired dashboard layout  

🚀 Deployment

The application is deployed using:
- Streamlit Cloud  

▶️ Run Locally

git clone https://github.com/your-username/diabetes-risk-predictor.git
cd diabetes-risk-predictor

pip install -r requirements.txt
streamlit run app.py

📦 Requirements

streamlit
numpy
pandas
scikit-learn
matplotlib

📁 Project Structure

diabetes-risk-predictor/
│
├── model/
│   ├── model.pkl
│   ├── scaler.pkl
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── data/
    └── diabetes.csv

🧪 Future Improvements

- Add ROC curve visualization  
- Add confusion matrix dashboard  
- Add patient history tracking  
- Deploy with database integration (PostgreSQL / Firebase)  
- Add multi-model comparison (XGBoost, SVM, etc.)  

👨‍⚕️ Disclaimer

This application is for educational purposes only and is not intended for real clinical diagnosis.

⭐ Author

Built as an AI/ML Capstone Project  
Focused on healthcare AI and clinical decision support systems