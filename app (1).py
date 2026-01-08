import streamlit as st
import pandas as pd
import joblib

# Load model (cached)
@st.cache_resource
def load_model():
    model = joblib.load("models/cardio_best_model_v1_fixed.pkl")
    return model

model = load_model()

st.title("🫀 Cardiovascular Disease Risk Prediction")
st.markdown("*Enter patient information to get instant risk assessment*")

# Sidebar inputs
st.sidebar.header("👤 Patient Information")

age_years = st.sidebar.slider("Age (years)", 18, 90, 50)
height = st.sidebar.slider("Height (cm)", 140, 210, 170)
weight = st.sidebar.slider("Weight (kg)", 40, 150, 75)
ap_hi = st.sidebar.slider("Systolic BP (mmHg)", 80, 240, 120)
ap_lo = st.sidebar.slider("Diastolic BP (mmHg)", 40, 200, 80)

bmi = weight / ((height / 100) ** 2)

age_group = st.sidebar.selectbox("Age Group", ["<40", "40-60", "60+"])
smoke = st.sidebar.selectbox("Smokes?", [0, 1])
alco = st.sidebar.selectbox("Alcohol?", [0, 1])
active = st.sidebar.selectbox("Physically active?", [0, 1])
cholesterol = st.sidebar.selectbox("Cholesterol", [1, 2, 3])
gluc = st.sidebar.selectbox("Glucose", [1, 2, 3])
gender = st.sidebar.selectbox("Gender (1=F, 2=M)", [1, 2])

high_ap_hi = 1 if ap_hi >= 140 else 0
high_ap_lo = 1 if ap_lo >= 90 else 0

# Predict button
if st.button("🔮 Predict Risk", type="primary"):
    # Input matching training X
    input_data = {
        "age_years": float(age_years),
        "height": float(height),
        "weight": float(weight),
        "ap_hi": float(ap_hi),
        "ap_lo": float(ap_lo),
        "bmi": float(bmi),
        "age_group": age_group,
        "smoke": int(smoke),
        "alco": int(alco),
        "active": int(active),
        "high_ap_hi": int(high_ap_hi),
        "high_ap_lo": int(high_ap_lo),
        "cholesterol": int(cholesterol),
        "gluc": int(gluc),
        "gender": int(gender),
    }
    
    input_df = pd.DataFrame([input_data])[list(X.columns)]
    
    # PREDICT
    risk_proba = model.predict_proba(input_df)[:, 1][0]
    prediction = int(risk_proba >= 0.5)
    
    # Display result
    st.subheader("📊 Prediction")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if prediction == 1:
            st.error("⚠️ **HIGH RISK**")
        else:
            st.success("✅ **LOW RISK**")
    
    with col2:
        st.metric("Disease Probability", f"{risk_proba:.1%}")
    
    st.write("### Input used:")
    st.dataframe(input_df)

st.markdown("---")
st.caption("Powered by scikit-learn Random Forest. Trained on cardiovascular dataset.")