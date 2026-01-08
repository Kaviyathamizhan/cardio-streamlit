
import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("models/cardio_best_model_v1.pkl")
    return model

model = load_model()

st.title("Cardiovascular Disease Risk Prediction")

st.write(
    "This app uses a trained machine learning model to predict the risk "
    "of cardiovascular disease based on patient information."
)

# Sidebar / input form
st.sidebar.header("Patient Information")

# Define inputs (adjust ranges as needed)
age_years = st.sidebar.slider("Age (years)", 18, 90, 50)
height = st.sidebar.slider("Height (cm)", 140, 210, 170)
weight = st.sidebar.slider("Weight (kg)", 40, 150, 75)
ap_hi = st.sidebar.slider("Systolic BP (ap_hi)", 80, 240, 120)
ap_lo = st.sidebar.slider("Diastolic BP (ap_lo)", 40, 200, 80)
bmi = weight / ((height / 100) ** 2)

age_group = st.sidebar.selectbox("Age Group", ["<40", "40-60", "60+"])

smoke = st.sidebar.selectbox("Smokes?", [0, 1])
alco = st.sidebar.selectbox("Alcohol intake?", [0, 1])
active = st.sidebar.selectbox("Physically active?", [0, 1])

high_ap_hi = 1 if ap_hi >= 140 else 0
high_ap_lo = 1 if ap_lo >= 90 else 0

cholesterol = st.sidebar.selectbox("Cholesterol (1=normal,2=above,3=high)", [1, 2, 3])
gluc = st.sidebar.selectbox("Glucose (1=normal,2=above,3=high)", [1, 2, 3])
gender = st.sidebar.selectbox("Gender (1=female,2=male)", [1, 2])

# Build input dict in same order as training X
input_data = {
    "age_years": age_years,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "bmi": bmi,
    "age_group": age_group,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "high_ap_hi": high_ap_hi,
    "high_ap_lo": high_ap_lo,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "gender": gender,
}

# Ensure we only use columns that exist in training X
# (in case you changed features)
all_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'age_group', 'smoke', 'alco', 'active', 'high_ap_hi', 'high_ap_lo', 'cholesterol', 'gluc', 'gender']
input_ordered = {c: input_data[c] for c in all_cols}

if st.button("Predict"):
    df = pd.DataFrame([input_ordered])
    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba >= 0.5)

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"High risk of cardiovascular disease (probability: {proba:.3f})")
    else:
        st.success(f"Low risk of cardiovascular disease (probability: {proba:.3f})")

    st.write("### Input summary")
    st.write(df)
    @st.cache_resource
def load_model():
    model = joblib.load("models/cardio_best_model_v1_fixed.pkl")  # ← CHANGED FILENAME
    return model
# ... rest same
