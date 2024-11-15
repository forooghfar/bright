import streamlit as st
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model and scaler parameters
model = load_model('dropout_model.h5')
scaler_mean = np.load('scaler.npy')
scaler_scale = np.load('scaler_scale.npy')
train_accuracy = np.load('train_accuracy.npy')
test_accuracy = np.load('test_accuracy.npy')

def predict_dropout(X_new):
    X_new = (X_new - scaler_mean) / scaler_scale
    # تغییر شکل دادن داده‌ها به شکل سه‌بعدی
    X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
    prediction = model.predict(X_new)[0][0]
    return prediction * 100

def recommended(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status):
    recommend = []
    if attendance_rate < 75:
        recommend.append("🔹 Encourage higher attendance by identifying any barriers the student may face.")
    if average_grade < 70:
        recommend.append("🔹 Provide academic support to improve grades.")
    if disciplinary_actions > 0:
        recommend.append("🔹 Work on behavioral guidance and mentoring to reduce disciplinary incidents.")
    if extracurricular_participation == 0:
        recommend.append("🔹 Encourage participation in extracurricular activities to increase engagement.")
    if parental_support == 0:
        recommend.append("🔹 Increase parental involvement through meetings, workshops, or counseling.")
    if economic_status == 1:
        recommend.append("🔹 Consider financial aid or support programs to alleviate economic impact.")
    return recommend

# Streamlit user interface
st.set_page_config(page_title="Dropout Prediction Students", page_icon="📚", layout="centered")
st.title("🔍 Student Dropout Prediction")

st.write(f"🔹 **Model Training Accuracy: {train_accuracy * 100:.2f}%**")
st.write(f"🔹 **Model Testing Accuracy: {test_accuracy * 100:.2f}%**")

# Input fields
st.subheader("📋 Student Info")
gender = st.selectbox('Gender', options=['Male', 'Female'])
gender = 1 if gender == 'Male' else 0
age = st.selectbox('Age', options=list(range(7, 19)))
attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 75)
average_grade = st.slider("Average Grade (%)", 0, 20, 15)
disciplinary_actions = st.number_input("Number of Disciplinary Incidents", min_value=0)
extracurricular_participation = st.selectbox("Extracurricular Participation", options=["No", "Yes"])
extracurricular_participation = 1 if extracurricular_participation == "Yes" else 0
parental_support = st.selectbox("Parental Support", options=["No", "Yes"])
parental_support = 1 if parental_support == "Yes" else 0
economic_status = st.selectbox("Economic Status", options=["Low", "Medium", "High"])
economic_status = {"Low": 1, "Medium": 2, "High": 3}[economic_status]

if st.button("Predict"):
    # Prepare input data
    new_data = np.array([[attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, age, gender, economic_status, parental_support]])
    dropout_risk = predict_dropout(new_data)
    st.write(f'🔹 **Dropout Risk: {dropout_risk:.2f}%**')

    # Recommendations based on dropout risk
    if dropout_risk > 50:
        st.subheader("📝 Suggested Recommendations:")
        recommendations = recommended(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status)
        for recommendation in recommendations:
            st.write(recommendation)
    else:
        st.write("🔹 **Student has a low risk of dropping out.**")

    