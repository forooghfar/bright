import streamlit as st
from app_dropout import run_dropout_app

# تنظیمات صفحه، اولین دستور در فایل اصلی
st.set_page_config(page_title="Main App", page_icon="📚", layout="centered")




# ایجاد منوی انتخاب
st.sidebar.title("انتخاب برنامه")
selection = st.sidebar.selectbox("برنامه مورد نظر خود را انتخاب کنید:", ["پیش‌بینی ترک تحصیل دانش‌آموزان"])

if selection == "پیش‌بینی ترک تحصیل دانش‌آموزان":
    run_dropout_app()
