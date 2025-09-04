import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Scores Predictor")

@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

st.title("Student Exam Scores – Predictor")
st.markdown("Enter the student's info and predict **math / reading / writing** scores.")

gender = st.selectbox("Gender", ["female", "male"])
race_ethnicity = st.selectbox("Race/Ethnicity", [
    "group A","group B","group C","group D","group E"
])
parental_level_of_education = st.selectbox("Parental level of education", [
    "some high school","high school","some college",
    "associate's degree","bachelor's degree","master's degree"
])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_prep = st.selectbox("Test preparation course", ["none", "completed"])

if st.button("Predict"):
    model = load_model()
    X_input = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race_ethnicity,
        "parental level of education": parental_level_of_education,
        "lunch": lunch,
        "test preparation course": test_prep
    }])
    preds = model.predict(X_input)[0]
    st.success(f"**Predicted scores** → Math: {preds[0]:.1f} | Reading: {preds[1]:.1f} | Writing: {preds[2]:.1f}")
