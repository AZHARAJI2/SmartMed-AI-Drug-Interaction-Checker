"""Streamlit entry point — role switcher for the two Phase 3 interfaces.

Run:  streamlit run ui/app.py
(the FastAPI backend must be running:  uvicorn main:app --port 8000)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path when launched via `streamlit run ui/app.py`
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from ui.doctor_page import render_doctor_page
from ui.patient_page import render_patient_page
from ui.widgets import inject_rtl

st.set_page_config(
    page_title="فاحص التفاعلات الدوائية بالذكاء الاصطناعي",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Arabic RTL layout
inject_rtl()

st.session_state.setdefault("api_base_url", "http://127.0.0.1:8000")
st.session_state.setdefault("token", "")
st.session_state.setdefault("email", "")

with st.sidebar:
    st.markdown("## 💊 فاحص التفاعلات الدوائية")
    role = st.radio("تسجيل الدخول بصفتك", ["مريض", "طبيب / صيدلاني"], key="role")
    st.divider()
    st.text_input("رابط خادم الـ API", key="api_base_url",
                  help="الرابط الأساسي لخادم FastAPI الخلفي (uvicorn main:app).")
    st.caption(f"توثيق الـ API التفاعلي (Swagger): [{st.session_state.api_base_url.rstrip('/')}/docs]({st.session_state.api_base_url.rstrip('/')}/docs)")

st.title("💊 فاحص التفاعلات الدوائية عبر صور العبوات")
st.caption("مساران متكاملان للتعرف على الدواء (قراءة النصوص OCR + مصنف الصور بنقل المعرفة) مع فحص التفاعلات عبر قاعدة بيانات DDInter ونظام التعلم التكيفي من تصحيحات الصيدلاني.")

if role == "مريض":
    render_patient_page()
else:
    render_doctor_page()
