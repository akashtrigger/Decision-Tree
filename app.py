import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Decision Tree Predictor",
    page_icon="🌳",
    layout="centered"
)

st.title("🌳 Decision Tree Purchase Prediction")
st.write("Predict whether a customer will **Purchase** or **Not Purchase**")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("decision_tree_model.pkl")

model = load_model()

# ---------------- USER INPUT ----------------
st.subheader("Enter Customer Details")

age = st.number_input("Age", min_value=18, max_value=70, value=30)
salary = st.number_input("Salary", min_value=20000, max_value=150000, step=1000)
experience = st.number_input("Experience (Years)", min_value=0, max_value=40, value=5)
education = st.selectbox(
    "Education Level",
    options=[0, 1, 2],
    format_func=lambda x: {0: "School", 1: "Undergraduate", 2: "Postgraduate"}[x]
)

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Salary": [salary],
        "Experience": [experience],
        "Education_Level": [education]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Prediction: Purchased")
    else:
        st.error("❌ Prediction: Not Purchased")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Decision Tree ML Model • Streamlit App")
