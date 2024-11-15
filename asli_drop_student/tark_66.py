import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import os
import io

# تنظیمات صفحه
st.set_page_config(page_title="Student Dropout Prediction", page_icon="📚", layout="centered")
st.title("🔍 Student Dropout Prediction")

# بارگذاری مدل و مقیاس‌کننده
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'dropout_model.h5')
scaler_mean_path = os.path.join(current_dir, 'scaler.npy')
scaler_scale_path = os.path.join(current_dir, 'scaler_scale.npy')

model = load_model(model_path)
scaler_mean = np.load(scaler_mean_path)
scaler_scale = np.load(scaler_scale_path)

# بارگذاری دقت آموزش و آزمون
train_accuracy_path = os.path.join(current_dir, 'train_accuracy.npy')
test_accuracy_path = os.path.join(current_dir, 'test_accuracy.npy')

try:
    train_accuracy = np.load(train_accuracy_path)
    test_accuracy = np.load(test_accuracy_path)
    st.write(f"🔹 **Model Training Accuracy: {train_accuracy * 100:.2f}%**")
    st.write(f"🔹 **Model Testing Accuracy: {test_accuracy * 100:.2f}%**")
except FileNotFoundError:
    st.warning("Training and test accuracy files not found.")

# تابع پیش‌بینی براساس ورودی جدید
def predict_dropout(X_new):
    X_new = (X_new - scaler_mean) / scaler_scale
    X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
    prediction = model.predict(X_new)[0][0]
    return prediction * 100

# توصیه‌ها براساس ورودی کاربر و اضافه کردن لینک‌های مفید
def recommended_with_links(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status):
    recommendations = []

    if attendance_rate < 75:
        recommendations.append((
            "🔹 Encourage higher attendance by identifying any barriers the student may face.",
            "[Learn More About Attendance Strategies](https://www.attendanceworks.org/)"
        ))

    if average_grade < 70:
        recommendations.append((
            "🔹 Provide academic support to improve grades.",
            "[Tutoring Resources](https://www.khanacademy.org/)"
        ))

    if disciplinary_actions > 0:
        recommendations.append((
            "🔹 Work on behavioral guidance and mentoring to reduce disciplinary incidents.",
            "[Behavioral Improvement Programs](https://www.coursera.org/learn/behavioral-psychology)"
        ))

    if extracurricular_participation == 0:
        recommendations.append((
            "🔹 Encourage participation in extracurricular activities to increase engagement.",
            "[Find Activities Near You](https://www.activityhero.com/)"
        ))

    if parental_support == 0:
        recommendations.append((
            "🔹 Increase parental involvement through meetings, workshops, or counseling.",
            "[Parent Engagement Tips](https://www.edutopia.org/family-engagement-strategies)"
        ))

    if economic_status == 1:
        recommendations.append((
            "🔹 Consider financial aid or support programs to alleviate economic challenges.",
            "[Explore Financial Aid](https://studentaid.gov/)"
        ))

    return recommendations

# دریافت اطلاعات کاربر
st.subheader("📋 Student Info")
gender = st.selectbox('Gender', options=['Male', 'Female'])
gender = 1 if gender == 'Male' else 0
age = st.selectbox('Age', options=list(range(7, 19)))
attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 75)
average_grade = st.slider("Average Grade (%)", 0, 100, 70)
disciplinary_actions = st.number_input("Number of Disciplinary Incidents", min_value=0)
extracurricular_participation = st.selectbox("Extracurricular Participation", options=["No", "Yes"])
extracurricular_participation = 1 if extracurricular_participation == "Yes" else 0
parental_support = st.selectbox("Parental Support", options=["No", "Yes"])
parental_support = 1 if parental_support == "Yes" else 0
economic_status = st.selectbox("Economic Status", options=["Low", "Medium", "High"])
economic_status = {"Low": 1, "Medium": 2, "High": 3}[economic_status]

# پیش‌بینی احتمال ترک تحصیل و نمایش توصیه‌ها
if st.button("Predict"):
    new_data = np.array([[attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, age, gender, economic_status, parental_support]])
    dropout_risk = predict_dropout(new_data)
    st.write(f'🔹 **Dropout Risk: {dropout_risk:.2f}%**')
    
    if dropout_risk > 50:
        st.subheader("📝 Suggested Recommendations:")
        recommendations = recommended_with_links(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status)
        for recommendation, link in recommendations:
            st.write(recommendation)
            st.markdown(link, unsafe_allow_html=True)
    else:
        st.write("🔹 **Student has a low risk of dropping out.**")

# تولید داده‌ها برای نمودارها
def generate_data(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status, age, gender):
    years = np.arange(age, age + 10)
    attendance_data = np.linspace(attendance_rate, attendance_rate - 5, 10)
    grade_data = np.linspace(average_grade, average_grade - 5, 10)
    disciplinary_data = np.random.randint(0, 3, 10)
    
    data = {
        'Age': years,
        'Attendance Rate': attendance_data,
        'Average Grade': grade_data,
        'Disciplinary Actions': disciplinary_data,
        'Extracurricular Participation': [extracurricular_participation] * 10,
        'Parental Support': [parental_support] * 10,
        'Economic Status': [economic_status] * 10
    }
    return pd.DataFrame(data)

df = generate_data(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status, age, gender)

# انتخاب نوع نمودار و نمایش آن
chart_type = st.selectbox("📊 Select Chart Type", options=["Line", "Bar", "Area", "Donut", "Scatter", "Comparison"], key='chart_type')
st.subheader("📈 Visualization")
fig = None

if chart_type == "Line":
    st.line_chart(df.set_index("Age")[["Average Grade", "Attendance Rate", "Disciplinary Actions"]])

elif chart_type == "Bar":
    st.bar_chart(df.set_index("Age")[["Average Grade", "Attendance Rate", "Disciplinary Actions"]])

elif chart_type == "Area":
    st.area_chart(df.set_index("Age")[["Average Grade", "Attendance Rate"]])

elif chart_type == "Donut":
    fig, ax = plt.subplots()
    values = [attendance_rate, average_grade, disciplinary_actions]
    labels = ["Attendance Rate", "Average Grade", "Disciplinary Actions"]
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
    plt.title("Donut Chart of Key Indicators")
    st.pyplot(fig)

elif chart_type == "Scatter":
    fig, ax = plt.subplots()
    sns.scatterplot(x="Age", y="Average Grade", data=df, ax=ax, label="Average Grade")
    sns.scatterplot(x="Age", y="Attendance Rate", data=df, ax=ax, label="Attendance Rate")
    ax.set_title("Scatter Plot of Average Grade and Attendance Rate Over Time")
    st.pyplot(fig)

elif chart_type == "Comparison":
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(x='Age', y='Average Grade', data=df, ax=axs[0], marker='o', label='Average Grade')
    sns.lineplot(x='Age', y='Attendance Rate', data=df, ax=axs[0], marker='o', label='Attendance Rate')
    axs[0].set_title("Comparison of Attendance and Grades")
    axs[0].set_xlabel("Age")
    axs[0].set_ylabel("Percentage")
    axs[0].legend()
    axs[1].fill_between(df['Age'], df['Attendance Rate'], alpha=0.3, label='Attendance Rate', color='green')
    axs[1].fill_between(df['Age'], df['Disciplinary Actions'], alpha=0.3, label='Disciplinary Actions', color='red')
    axs[1].set_title("Area Comparison of Attendance and Disciplinary Actions")
    axs[1].set_xlabel("Age")
    axs[1].legend()
    st.pyplot(fig)

# دکمه دانلود نمودار
if fig:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="📥 Download Chart",
        data=buf,
        file_name="chart.png",
        mime="image/png"
    )
