import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("AI-Based Clinical Decision Support System for Early Cardiovascular Risk Prediction")

st.title("🫀 AI-Based Early Heart Disease Prediction System")
st.markdown("### Research Dashboard - Random Forest Model")

# Sidebar for patient input
st.sidebar.header("Enter Patient Details")

# Age
age = st.sidebar.number_input("Age", 20, 100, 50)

# Sex
sex_option = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_option == "Male" else 0

# Chest Pain Type
cp_option = st.sidebar.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", 
     "Non-anginal Pain", "Asymptomatic"]
)

cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

cp = cp_mapping[cp_option]

# Resting Blood Pressure
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)

# Cholesterol
chol = st.sidebar.number_input("Cholesterol Level", 100, 400, 200)

# Fasting Blood Sugar
fbs_option = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"]
)
fbs = 1 if fbs_option == "Yes" else 0

# Rest ECG
restecg = st.sidebar.selectbox("Rest ECG Result (0-2)", [0,1,2])

# Max Heart Rate
thalach = st.sidebar.number_input("Maximum Heart Rate", 60, 220, 150)

# Exercise Induced Angina
exang_option = st.sidebar.selectbox(
    "Exercise Induced Angina", ["No", "Yes"]
)
exang = 1 if exang_option == "Yes" else 0

# ST Depression
oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

# Slope
slope = st.sidebar.selectbox("Slope of ST Segment", [0,1,2])

# Number of Major Vessels
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])

# Thalassemia
thal_option = st.sidebar.selectbox(
    "Thalassemia Type",
    ["Normal", "Fixed Defect", "Reversible Defect"]
)

thal_mapping = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

thal = thal_mapping[thal_option]

# Collect input into dataframe
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

scaled_data = scaler.transform(input_data)

# Prediction button
if st.sidebar.button("Predict"):

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("Prediction Result")

    risk_percent = round(probability * 100, 2)

    if risk_percent < 40:
        st.info("Clinical Insight: Patient shows low cardiovascular risk based on current parameters.")
    elif risk_percent < 70:
        st.warning("Clinical Insight: Moderate cardiovascular risk detected. Lifestyle modification recommended.")
    else:
        st.error("Clinical Insight: High cardiovascular risk detected. Immediate medical consultation advised.")

    st.markdown("---")
    st.subheader("Model Confidence")
    st.write(f"Probability of Heart Disease: {risk_percent}%")

    # Feature Importance Visualization
    st.markdown("---")
    st.subheader("Feature Importance (Model Insight)")

    importances = model.feature_importances_
    features = ["age","sex","cp","trestbps","chol","fbs","restecg",
                "thalach","exang","oldpeak","slope","ca","thal"]

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

st.markdown("---")
st.markdown(
    "🔬 Developed by Aaman Shaikh | "
    "AI Healthcare Research Project | "
    "Random Forest Classifier | 98.5% Accuracy"
)

