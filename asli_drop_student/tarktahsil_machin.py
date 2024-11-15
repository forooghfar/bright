import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# بارگذاری داده‌ها
df = pd.read_csv('student_data.csv')

# تعریف ویژگی‌ها و هدف
X = df.drop('dropped_out', axis=1)
y = df['dropped_out']

# Encode categorical features
features = ['attendance_rate', 'average_grade', 'disciplinary_actions', 'extracurricular_participation', 
            'age', 'gender', 'economic_status', 'parental_support']
for feature in features:
    coder = LabelEncoder()
    X[feature] = coder.fit_transform(X[feature])

# تقسیم داده‌ها به مجموعه آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تعریف مدل و آموزش آن
rf_model = RandomForestClassifier(n_estimators=100, random_state=38)
rf_model.fit(X_train, y_train)

# محاسبه دقت مدل
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

# تابع پیش‌بینی
def predict_dropout(X_new):
    prediction = rf_model.predict_proba(X_new)[0][1]
    return prediction * 100

# توصیه‌ها
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

# رابط کاربری Streamlit
def main():
    st.set_page_config(page_title="Dropout Prediction Students", page_icon="📚", layout="centered")
    st.title("🔍 Student Dropout Prediction")
    
    # نمایش دقت مدل
    st.write(f"🔹 **Model Training Accuracy: {train_accuracy * 100:.2f}%**")
    st.write(f"🔹 **Model Testing Accuracy: {test_accuracy * 100:.2f}%**")

    # Input fields
    st.subheader("📋 Student Info")
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    gender = 1 if gender == 'Male' else 0
    age = st.slider('Age', min_value=15, max_value=20, value=17)
    attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 75)
    average_grade = st.slider("Average Grade (%)", 0, 100, 70)
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

        # Show recommendations if risk is high
        if dropout_risk > 50:
            st.subheader("📝 Suggested Recommendations:")
            recommendations = recommended(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status)
            for recommendation in recommendations:
                st.write(recommendation)
        else:
            st.write("🔹 **Student has a low risk of dropping out.**")

        # Feature importance plot
        st.subheader("📊 Feature Importance for Dropout Prediction")
        importances = rf_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = importances[sorted_indices]

        fig, ax = plt.subplots()
        ax.pie(sorted_importances[:5], labels=sorted_features[:5], autopct='%1.2f%%', startangle=90)
        ax.axis('equal') 
        st.pyplot(fig)

if __name__ == "__main__":
    main()
