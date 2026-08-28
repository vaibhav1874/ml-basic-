import streamlit as st
import pandas as pd
import joblib


model = joblib.load("svm_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_col = joblib.load("col.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Predictor")
st.markdown("Please provide the patient details below to predict the risk of heart disease.")


col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=20, max_value=100, value=50, step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=["ASY", "ATA", "NAP", "TA"],
        help="ASY: Asymptomatic | ATA: Atypical Angina | NAP: Non-Anginal Pain | TA: Typical Angina"
    )
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)
    cholesterol = st.slider("Serum Cholesterol (mm/dl)", min_value=100, max_value=600, value=200, step=1)
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"]
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        options=["Normal", "ST", "LVH"],
        help="Normal | ST: ST-T wave abnormality | LVH: Left ventricular hypertrophy"
    )
    max_hr = st.slider("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=140, step=1)
    exercise_angina = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
    oldpeak = st.slider("Oldpeak (ST Depression)", min_value=-2.5, max_value=6.5, value=0.0, step=0.1)
    st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

st.markdown("---")

if st.button("🔍 Predict Risk", use_container_width=True):
    
    is_female = 1 if sex == "Female" else 0
    is_exercise_angina = 1 if exercise_angina == "Yes" else 0
    is_up = 1 if st_slope == "Up" else 0
    f_bs = 1 if fasting_bs.startswith("Yes") else 0

   
    bp_normal = 1 if resting_bp <= 120 else 0
    bp_elevated = 1 if 120 < resting_bp <= 130 else 0
    bp_high = 1 if 130 < resting_bp <= 140 else 0
    bp_very_high = 1 if resting_bp > 140 else 0

   
    input_data = {
        'Age': age,
        'is_female': is_female,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': f_bs,
        'MaxHR': max_hr,
        'is_ExerciseAngina': is_exercise_angina,
        'Oldpeak': oldpeak,
        'is_up': is_up,
        'ChestPainType_ASY': 1 if chest_pain == "ASY" else 0,
        'ChestPainType_ATA': 1 if chest_pain == "ATA" else 0,
        'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,
        'ChestPainType_TA': 1 if chest_pain == "TA" else 0,
        'RestingECG_LVH': 1 if resting_ecg == "LVH" else 0,
        'RestingECG_Normal': 1 if resting_ecg == "Normal" else 0,
        'RestingECG_ST': 1 if resting_ecg == "ST" else 0,
        'RestingBP_cat_Normal': bp_normal,
        'RestingBP_cat_Elevated': bp_elevated,
        'RestingBP_cat_High': bp_high,
        'RestingBP_cat_Very High': bp_very_high
    }

  
    df_input = pd.DataFrame([input_data])[expected_col]


    df_scaled = scaler.transform(df_input)
    prediction = model.predict(df_scaled)[0]

   
    if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease Detected**\nPlease consult a medical professional for further clinical evaluation.")
    else:
        st.success("✅ **Low Risk / Normal**\nNo strong indications of heart disease based on the provided inputs.")
