import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns






import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import Sequential    
from tensorflow.keras.models import load_model
import pandas as pd

# بارگذاری مدل و مقیاس‌کننده
model = load_model('dropout_model.h5')
scaler_mean = np.load('scaler.npy')
scaler_scale = np.load('scaler_scale.npy')
train_accuracy = np.load('D:/AL/challange/a_code_asli/asli_drop_student/train_accuracy.npy')
test_accuracy = np.load('D:/AL/challange/a_code_asli/asli_drop_student/test_accuracy.npy')

def predict_dropout(X_new):
    X_new = (X_new - scaler_mean) / scaler_scale
    X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
    prediction = model.predict(X_new)[0][0]
    return prediction * 100

def recommended(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status):
    recommend = []
    
    if attendance_rate < 75:
        recommend.append("🔹 Encourage higher attendance by identifying any barriers the student may face.")
        recommend.append("[Video: Motivation for Regular School Attendance](https://www.youtube.com/watch?v=1qWwIIt1aIE)")
        recommend.append("[Book: Effective Attendance Improvement Strategies](https://www.amazon.com/Improving-School-Attendance-Michael-Reynolds/dp/1475852805)")

    if average_grade < 70:
        recommend.append("🔹 Provide academic support to improve grades.")
        recommend.append("[Book: Academic Success for Struggling Students](https://www.amazon.com/Strategies-Successful-Students-Michael-Reynolds/dp/1475852740)")

    if disciplinary_actions > 0:
        recommend.append("🔹 Work on behavioral guidance and mentoring to reduce disciplinary incidents.")
        recommend.append("[Workshop: Behavioral Improvement Workshop](https://www.coursera.org/learn/behavioral-psychology)")

    if extracurricular_participation == 0:
        recommend.append("🔹 Encourage participation in extracurricular activities to increase engagement.")
        recommend.append("[Video: Study Tips and Techniques](https://www.youtube.com/watch?v=3ggm4w-wvTY)")

    if parental_support == 0:
        recommend.append("🔹 Increase parental involvement through meetings, workshops, or counseling.")
        recommend.append("[Workshop: Parent Engagement Strategies](https://www.edutopia.org/family-engagement-strategies)")

    if economic_status == 1:
        recommend.append("🔹 Consider financial aid or support programs to alleviate economic impact.")
        recommend.append("[Course: Financial Aid and Support Resources](https://www.coursera.org/learn/financial-aid)")

    return recommend

#streamlit user interface
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








# # فرض کنید این داده‌ها بر اساس ورودی‌های کاربر در قسمت پیش‌بینی وارد شده‌اند
# def generate_data(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status, age, gender):
#     # تولید داده‌های مشابه برای 10 سال بر اساس ورودی‌های کاربر
#     years = np.arange(age, age + 10)
#     attendance_data = np.linspace(attendance_rate, attendance_rate - 5, 10)
#     grade_data = np.linspace(average_grade, average_grade - 5, 10)
#     disciplinary_data = np.random.randint(0, 3, 10)  # ایجاد داده‌های تصادفی برای رفتار انضباطی

#     # ساخت DataFrame برای نمایش گراف‌ها
#     data = {
#         'Age': years,
#         'Attendance Rate': attendance_data,
#         'Average Grade': grade_data,
#         'Disciplinary Actions': disciplinary_data,
#         'Extracurricular Participation': [extracurricular_participation] * 10,
#         'Parental Support': [parental_support] * 10,
#         'Economic Status': [economic_status] * 10
#     }
#     return pd.DataFrame(data)

# # دریافت ورودی‌های کاربر
# st.subheader("📋 Student Info")
# age = st.selectbox('Age', options=list(range(7, 19)))
# attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 75)
# average_grade = st.slider("Average Grade (%)", 0, 20, 15)
# disciplinary_actions = st.number_input("Number of Disciplinary Incidents", min_value=0)
# extracurricular_participation = st.selectbox("Extracurricular Participation", options=["No", "Yes"])
# extracurricular_participation = 1 if extracurricular_participation == "Yes" else 0
# parental_support = st.selectbox("Parental Support", options=["No", "Yes"])
# parental_support = 1 if parental_support == "Yes" else 0
# economic_status = st.selectbox("Economic Status", options=["Low", "Medium", "High"])
# economic_status = {"Low": 1, "Medium": 2, "High": 3}[economic_status]
# gender = st.selectbox('Gender', options=['Male', 'Female'])
# gender = 1 if gender == 'Male' else 0

# # تولید داده‌ها
# df = generate_data(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status, age, gender)

# # انتخاب نوع نمودار توسط کاربر
# chart_type = st.selectbox("📊 Select Chart Type", options=["Line", "Bar", "Area", "Donut", "Scatter", "Comparison"])

# # نمایش گراف‌ها بر اساس نوع انتخاب‌شده توسط کاربر
# st.subheader("📈 Visualization")

# if chart_type == "Line":
#     st.line_chart(df.set_index("Age")[["Average Grade", "Attendance Rate", "Disciplinary Actions"]])

# elif chart_type == "Bar":
#     st.bar_chart(df.set_index("Age")[["Average Grade", "Attendance Rate", "Disciplinary Actions"]])

# elif chart_type == "Area":
#     st.area_chart(df.set_index("Age")[["Average Grade", "Attendance Rate"]])

# elif chart_type == "Donut":
#     fig, ax = plt.subplots()
#     values = [attendance_rate, average_grade, disciplinary_actions]
#     labels = ["Attendance Rate", "Average Grade", "Disciplinary Actions"]
#     ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
#     plt.title("Donut Chart of Key Indicators")
#     st.pyplot(fig)

# elif chart_type == "Scatter":
#     fig, ax = plt.subplots()
#     sns.scatterplot(x="Age", y="Average Grade", data=df, ax=ax, label="Average Grade")
#     sns.scatterplot(x="Age", y="Attendance Rate", data=df, ax=ax, label="Attendance Rate")
#     ax.set_title("Scatter Plot of Average Grade and Attendance Rate Over Time")
#     st.pyplot(fig)

# elif chart_type == "Comparison":
#     fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
#     # نمودار مقایسه‌ای برای حضور و نمرات
#     sns.lineplot(x='Age', y='Average Grade', data=df, ax=axs[0], marker='o', label='Average Grade')
#     sns.lineplot(x='Age', y='Attendance Rate', data=df, ax=axs[0], marker='o', label='Attendance Rate')
#     axs[0].set_title("Comparison of Attendance and Grades")
#     axs[0].set_xlabel("Age")
#     axs[0].set_ylabel("Percentage")
#     axs[0].legend()

#     # نمودار مساحتی برای حضور و رفتارهای انضباطی
#     axs[1].fill_between(df['Age'], df['Attendance Rate'], alpha=0.3, label='Attendance Rate', color='green')
#     axs[1].fill_between(df['Age'], df['Disciplinary Actions'], alpha=0.3, label='Disciplinary Actions', color='red')
#     axs[1].set_title("Area Comparison of Attendance and Disciplinary Actions")
#     axs[1].set_xlabel("Age")
#     axs[1].legend()
    
#     st.pyplot(fig)


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# فرض کنید این داده‌ها بر اساس ورودی‌های کاربر در قسمت پیش‌بینی وارد شده‌اند
def generate_data(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status, age, gender):
    # تولید داده‌های مشابه برای 10 سال بر اساس ورودی‌های کاربر
    years = np.arange(age, age + 10)
    attendance_data = np.linspace(attendance_rate, attendance_rate - 5, 10)
    grade_data = np.linspace(average_grade, average_grade - 5, 10)
    disciplinary_data = np.random.randint(0, 3, 10)  # ایجاد داده‌های تصادفی برای رفتار انضباطی

    # ساخت DataFrame برای نمایش گراف‌ها
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

# # دریافت ورودی‌های کاربر
# st.subheader("📋 Student Info")
# age = st.selectbox('Age', options=list(range(7, 19)), key='age_selectbox')
# attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 75, key='attendance_slider')
# average_grade = st.slider("Average Grade (%)", 0, 20, 15, key='grade_slider')
# disciplinary_actions = st.number_input("Number of Disciplinary Incidents", min_value=0, key='disciplinary_input')
# extracurricular_participation = st.selectbox("Extracurricular Participation", options=["No", "Yes"], key='extra_participation')
# extracurricular_participation = 1 if extracurricular_participation == "Yes" else 0
# parental_support = st.selectbox("Parental Support", options=["No", "Yes"], key='parental_support')
# parental_support = 1 if parental_support == "Yes" else 0
# economic_status = st.selectbox("Economic Status", options=["Low", "Medium", "High"], key='economic_status')
# economic_status = {"Low": 1, "Medium": 2, "High": 3}[economic_status]
# gender = st.selectbox('Gender', options=['Male', 'Female'], key='gender')
# gender = 1 if gender == 'Male' else 0

# تولید داده‌ها
df = generate_data(attendance_rate, average_grade, disciplinary_actions, extracurricular_participation, parental_support, economic_status, age, gender)

# انتخاب نوع نمودار توسط کاربر
chart_type = st.selectbox("📊 Select Chart Type", options=["Line", "Bar", "Area", "Donut", "Scatter", "Comparison"], key='chart_type')

# نمایش گراف‌ها بر اساس نوع انتخاب‌شده توسط کاربر
st.subheader("📈 Visualization")

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
    
    # نمودار مقایسه‌ای برای حضور و نمرات
    sns.lineplot(x='Age', y='Average Grade', data=df, ax=axs[0], marker='o', label='Average Grade')
    sns.lineplot(x='Age', y='Attendance Rate', data=df, ax=axs[0], marker='o', label='Attendance Rate')
    axs[0].set_title("Comparison of Attendance and Grades")
    axs[0].set_xlabel("Age")
    axs[0].set_ylabel("Percentage")
    axs[0].legend()

    # نمودار مساحتی برای حضور و رفتارهای انضباطی
    axs[1].fill_between(df['Age'], df['Attendance Rate'], alpha=0.3, label='Attendance Rate', color='green')
    axs[1].fill_between(df['Age'], df['Disciplinary Actions'], alpha=0.3, label='Disciplinary Actions', color='red')
    axs[1].set_title("Area Comparison of Attendance and Disciplinary Actions")
    axs[1].set_xlabel("Age")
    axs[1].legend()
    
    st.pyplot(fig)
