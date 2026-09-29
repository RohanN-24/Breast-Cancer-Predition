import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page setup
st.set_page_config(
    page_title=" Breast Cancer Risk Assessment",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
        /* Main background & headers */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #E91E63, #9C27B0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #6c757d;
            margin-bottom: 1.5rem;
        }
        
        /* Card-like containers */
        .category-card {
            background: rgba(255, 255, 255, 0.05);
            border-left: 4px solid #E91E63;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 12px;
            font-weight: 600;
            font-size: 1.1rem;
        }

        /* Result styling */
        .result-box-positive {
            background: linear-gradient(135deg, rgba(233, 30, 99, 0.15), rgba(244, 67, 54, 0.25));
            border: 1px solid #E91E63;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .result-box-negative {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(0, 150, 136, 0.25));
            border: 1px solid #4CAF50;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-top: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# 1. Load pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("cancer_prediction_pipeline.pkl")

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("Error: `cancer_prediction_pipeline.pkl` not found. Please run the training script first.")
    st.stop()

# Header Section
st.markdown('<div class="main-header">🎗️ Breast Cancer AI Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Clinical decision support tool powered by Gradient Boosting.</div>', unsafe_allow_html=True)

# Sidebar with quick info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/pink-ribbon.png", width=64)
    st.subheader("About Model")
    st.info("""
    - **Model:** Gradient Boosting Classifier
    - **Dataset Size:** 10,000 Patient Records
    - **Key Indicators:** Tumor size, mammogram results, age, BMI & genetic predisposition.
    """)
    st.caption("⚠️ Note: This tool provides predictive estimates for educational/research purposes and is not a substitute for a professional biopsy.")

# Main Input Form
with st.form("diagnostic_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="category-card">🩺 Patient Vitals</div>', unsafe_allow_html=True)
        age = st.slider("Age", min_value=18, max_value=90, value=48)
        gender = st.selectbox("Gender", ["Female", "Male"])
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=55.0, value=26.5, step=0.1)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120, step=1)
        cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=350, value=200, step=1)

    with col2:
        st.markdown('<div class="category-card">🔬 Diagnostic & Exam</div>', unsafe_allow_html=True)
        tumor_size = st.number_input("Tumor Size (cm)", min_value=0.0, max_value=12.0, value=2.1, step=0.1)
        mammogram = st.selectbox("Mammogram Finding", ["Normal", "Suspicious", "Benign"])
        lymph_node = st.selectbox("Lymph Node Involvement", ["No", "Yes"])
        menopause = st.selectbox("Menopause Status", ["Pre", "Post"])
        breastfeeding = st.selectbox("Breastfeeding History", ["Yes", "No"])
        genetic_mutation = st.selectbox("Genetic Mutation (BRCA)", ["Negative", "Positive"])

    with col3:
        st.markdown('<div class="category-card">🏃 Lifestyle & History</div>', unsafe_allow_html=True)
        family_history = st.selectbox("Family History of Cancer", ["No", "Yes"])
        smoking = st.selectbox("Smoking Habits", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        hormone_therapy = st.selectbox("Hormone Replacement Therapy", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        physical_activity = st.selectbox("Physical Activity", ["Moderate", "High", "Low"])
        exercise_days = st.slider("Weekly Exercise Days", min_value=0, max_value=7, value=3)

    submitted = st.form_submit_button("🔍 Run Cancer Risk Analysis", use_container_width=True)

# Prediction Result
if submitted:
    # Compute Risk Factor Count
    risk_factors = [
        1 if family_history == "Yes" else 0,
        1 if smoking == "Yes" else 0,
        1 if alcohol == "Yes" else 0,
        1 if hormone_therapy == "Yes" else 0,
        1 if genetic_mutation == "Positive" else 0,
        1 if diabetes == "Yes" else 0,
    ]
    risk_factor_count = sum(risk_factors)

    # Input DataFrame (without Annual_Income_USD)
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "Family_History": family_history,
        "Smoking": smoking,
        "Alcohol_Consumption": alcohol,
        "Physical_Activity": physical_activity,
        "Hormone_Therapy": hormone_therapy,
        "Menopause_Status": menopause,
        "Genetic_Mutation": genetic_mutation,
        "Tumor_Size_cm": tumor_size,
        "Lymph_Node_Involvement": lymph_node,
        "Mammogram_Result": mammogram,
        "Blood_Pressure": blood_pressure,
        "Cholesterol": cholesterol,
        "Diabetes": diabetes,
        "Exercise_Days_Per_Week": exercise_days,
        "Breastfeeding_History": breastfeeding,
        "Annual_Income_USD": 82000,
        "Risk_Factor_Count": float(risk_factor_count)
    }])

    # Run Prediction
    prediction = pipeline.predict(input_df)[0]
    prob_malignant = pipeline.predict_proba(input_df)[0][1]

    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1.2, 1.8])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
                <div class="result-box-positive">
                    <h3 style="color: #E91E63; margin-bottom: 4px;">⚠️ High Risk (Malignant)</h3>
                    <p style="font-size: 0.95rem; color: #ced4da;">Patient parameters strongly align with positive cancer cases.</p>
                    <span class="risk-badge" style="background-color: #E91E63; color: white;">Action Recommended</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box-negative">
                    <h3 style="color: #4CAF50; margin-bottom: 4px;">✅ Low Risk (Benign / Normal)</h3>
                    <p style="font-size: 0.95rem; color: #ced4da;">Current markers suggest low probability of cancer.</p>
                </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.subheader("Confidence & Diagnostics")
        st.write(f"**Predicted Malignancy Probability:** `{prob_malignant * 100:.2f}%`")
        
        # Color bar indicator
        st.progress(float(prob_malignant))
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Cumulative Risk Factors", f"{risk_factor_count} / 6")
        with c2:
            st.metric("Tumor Size Checked", f"{tumor_size} cm")