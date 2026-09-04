"""صفحة المريض: تسجيل/دخول، إدارة الأدوية مع فحص تعارض تلقائي، مسح العبوة، سجل الفحوصات."""
from __future__ import annotations

import streamlit as st

from ui.api_client import ApiClient, ApiError
from ui.pdf_report import generate_patient_pdf_report
from ui.widgets import COMMON_NAMES_MAP, REVIEW_STATUS_AR, render_report, render_scan_view


def _client() -> ApiClient:
    return ApiClient(st.session_state.api_base_url, token=st.session_state.get("token", ""))


def _auth_forms() -> None:
    st.subheader("دخول المريض")
    tab_login, tab_register = st.tabs(["تسجيل الدخول", "حساب جديد"])
    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("البريد الإلكتروني", key="login_email")
            password = st.text_input("كلمة المرور", type="password", key="login_password")
            if st.form_submit_button("دخول", type="primary"):
                try:
                    st.session_state.token = _client().login(email, password)
                    st.session_state.email = email
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
    with tab_register:
        with st.form("register_form", clear_on_submit=False):
            name = st.text_input("الاسم الكامل", key="reg_name")
            email = st.text_input("البريد الإلكتروني", key="reg_email")
            password = st.text_input("كلمة المرور (6 أحرف على الأقل)", type="password", key="reg_password")
            if st.form_submit_button("إنشاء حساب", type="primary"):
                try:
                    st.session_state.token = _client().register(name, email, password)
                    st.session_state.email = email
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))


def _auto_interaction_check(client: ApiClient, names: list[str], medications: list[dict] | None = None) -> None:
    """الفحص التلقائي: دواء واحد عادي، وأكثر من دواء يُفحصون تلقائيًا ضد DDInter."""
    report = None
    if len(names) <= 1:
        if names:
            st.success(f"أنت تتناول حاليًا دواءً واحدًا (**{names[0]}**) — لا يوجد أي احتمال "
                       "لتعارض دوائي. سيبدأ الفحص التلقائي فور إضافة دواءٍ ثانٍ.")
        else:
            st.info("قائمتك فارغة حاليًا. أضف أول دواء لبدء المتابعة والفحص التلقائي.")
    else:
        with st.spinner(f"جارٍ فحص التعارض بين أدويتك الـ{len(names)} تلقائيًا..."):
            try:
                report = client.doctor_check(names)
            except ApiError as exc:
                st.error(str(exc))
                return
        st.write(f"تم فحص **{len(names)}** أدوية من قائمتك تلقائيًا:")
        render_report(report)

    if medications:
        st.divider()
        st.subheader("📄 التقرير الطبي الشامل (PDF)")
        st.caption("يمكنك تحميل تقرير طبي منسق بصيغة PDF يوضح قائمتك الحالية ونتائج فحص التفاعلات لتقديمه لطبيبك المعالج.")
        try:
            pdf_bytes = generate_patient_pdf_report(
                patient_email=st.session_state.get("email", "Patient"),
                medications=medications,
                interaction_report=report,
            )
            email_slug = st.session_state.get("email", "patient").split("@")[0]
            st.download_button(
                label="📥 تحميل التقرير الطبي (PDF)",
                data=pdf_bytes,
                file_name=f"Medication_Report_{email_slug}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as err:
            st.warning(f"تعذر تجهيز التقرير: {err}")


def _clean_drug_name(name: str) -> str:
    from config import CONFIG
    cleaned = name.strip()
    for junk in CONFIG.split.junk_keywords:
        if junk in cleaned.lower():
            return ""
    return cleaned


def _render_medications(client: ApiClient) -> None:
    st.write("أضف أي دواء تتناوله بالاسم التجاري أو العلمي، وسيقوم النظام "
             "بفحص التعارض بينها **تلقائيًا** فور الإضافة أو الحذف.")

    medications = client.list_medications()
    names = [m["drug_name"] for m in medications]

    # ── Current medication list ──────────────────────────────────────────────
    if not medications:
        st.info("💡 قائمتك فارغة حاليًا. أضف أول دواء أدناه أو صوّر العبوة من تبويب "
                "**(📷 فحص عبوة دواء)**.")
    else:
        st.markdown(f"**قائمة أدويتك الحالية ({len(medications)} أدوية):**")
        for med in medications:
            col_name, col_remove = st.columns([5, 1])
            friendly_name = COMMON_NAMES_MAP.get(med["drug_name"].lower(), med["drug_name"])
            col_name.markdown(f"💊 **{friendly_name}**")
            if col_remove.button("❌ حذف", key=f"pat_rm_{med['drug_id']}"):
                client.remove_medication(med["drug_id"])
                st.rerun()

    st.markdown("---")
    st.markdown("#### ➕ إضافة دواء جديد")

    # ── Option A: type any name directly ────────────────────────────────────
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        direct_name = st.text_input(
            "اكتب اسم الدواء مباشرة (تجاري أو علمي — مثال: Aspirin، بنادول، Metformin):",
            key="pat_direct_name",
            placeholder="مثال: Aspirin أو Panadol أو Metformin",
        )
    with col_btn:
        st.write("")
        st.write("")
        add_direct = st.button("➕ إضافة", key="pat_add_direct", type="primary")

    if add_direct and direct_name.strip():
        clean = _clean_drug_name(direct_name)
        if not clean:
            st.warning("الاسم يبدو غير صالح، يرجى التأكد منه.")
        else:
            try:
                added = client.add_medication(clean)
                st.success(f"✅ تمت إضافة **{added['drug_name']}** إلى قائمتك بنجاح!")
                st.rerun()
            except ApiError as exc:
                st.error(f"تعذرت الإضافة: {exc}")

    # ── Option B: search and pick from DB ───────────────────────────────────
    with st.expander("🔍 بحث في قاعدة البيانات للحصول على الاسم الدقيق"):
        query = st.text_input("ابحث عن دواء بالاسم الإنجليزي", key="pat_med_query",
                              placeholder="مثال: paracetamol أو dolutegravir")
        if query:
            candidates = client.search_drugs(query)
            if not candidates:
                st.warning("لا يوجد دواء مطابق بهذا الاسم في قاعدة البيانات. يمكنك إضافته مباشرة بالخانة أعلاه.")
            else:
                options = {c["trade_name"]: c["drug_id"] for c in candidates}
                choice = st.selectbox("اختر الدواء:", list(options), key="pat_med_choice")
                if st.button("➕ إضافة الدواء المختار", type="primary", key="pat_add_choice"):
                    try:
                        added = client.add_medication(choice)
                        st.success(f"✅ تم حفظ **{added['drug_name']}** في قائمتك.")
                        st.rerun()
                    except ApiError as exc:
                        st.error(str(exc))

    st.divider()
    st.subheader("🔬 الفحص التلقائي للتعارض الدوائي")
    _auto_interaction_check(client, names, medications)



def _render_scan(client: ApiClient) -> None:
    st.write("التقط صورة بالكاميرا أو ارفع صورة لعبوة الدواء؛ سيتعرف عليها النظام "
             "(قراءة النص + تصنيف الصورة)، يفحص التفاعلات، ويتيح لك حفظه في قائمة أدويتك فورًا.")

    input_mode = st.radio(
        "طريقة إدخال الصورة",
        ["📸 تصوير مباشر بالكاميرا", "📁 رفع صورة من الجهاز"],
        horizontal=True,
        key="patient_input_mode",
    )

    image_bytes = None
    image_name = "scan.jpg"

    if input_mode == "📸 تصوير مباشر بالكاميرا":
        camera_img = st.camera_input("وجّه الكاميرا نحو علبة أو شريط الدواء بوضوح", key="patient_camera")
        if camera_img is not None:
            image_bytes = camera_img.getvalue()
            image_name = "camera_capture.jpg"
    else:
        upload = st.file_uploader("اختر صورة عبوة الدواء", type=["jpg", "jpeg", "png"], key="patient_scan")
        if upload is not None:
            image_bytes = upload.getvalue()
            image_name = upload.name

    if image_bytes is not None and st.button("🔍 فحص الصورة الآن", type="primary", key="btn_patient_scan"):
        with st.spinner("جارٍ الفحص (قراءة النص + تصنيف الصورة + فحص التفاعلات)..."):
            try:
                view = client.scan_image(image_bytes, image_name)
            except ApiError as exc:
                st.error(str(exc))
                return

        render_scan_view(view, show_ocr=True)

        if view.accepted:
            candidate = view.matched_drug_name or view.fused_label or view.ocr_label
            detected = _clean_drug_name(candidate)
            if detected:
                st.markdown("---")
                friendly = COMMON_NAMES_MAP.get(detected.lower(), detected)
                st.success(f"✅ تم التعرف على الدواء: **{friendly}** (`{detected}`)")
                if st.button(f"➕ إضافة {friendly} إلى قائمة أدويتي الدائمة", type="primary", key="btn_add_to_my_meds"):
                    try:
                        client.add_medication(detected)
                        st.success(f"تم حفظ **{friendly}** في قائمة أدويتك بنجاح!")
                        st.rerun()
                    except ApiError as exc:
                        st.error(f"تعذر الحفظ: {exc}")


def _render_history(client: ApiClient) -> None:
    history = client.scan_history()
    if not history:
        st.info("لا توجد فحوصات بعد — جرّب تبويب *فحص عبوة دواء*.")
        return
    st.dataframe([{
        "رقم الفحص": h["scan_id"],
        "الصورة": h["image_path"],
        "تصنيف الصورة": h["classifier_prediction"] or "—",
        "الثقة": f"{h['fused_confidence']:.0%}",
        "الدواء المطابق": h["matched_drug_id"] or "—",
        "حالة المراجعة": REVIEW_STATUS_AR.get(h["review_status"], h["review_status"]),
        "المراجع": h["reviewed_by"] or "—",
    } for h in history], use_container_width=True, hide_index=True)


def render_patient_page() -> None:
    if not st.session_state.get("token"):
        _auth_forms()
        return
    top = st.columns([4, 1])
    top[0].success(f"مرحبًا **{st.session_state.get('email', '')}** 👋")
    if top[1].button("تسجيل الخروج"):
        st.session_state.token = ""
        st.rerun()
    client = _client()
    tabs = st.tabs(["💊 أدويتي والفحص التلقائي", "📷 فحص عبوة دواء", "🗂️ سجل الفحوصات"])
    with tabs[0]:
        _render_medications(client)
    with tabs[1]:
        _render_scan(client)
    with tabs[2]:
        _render_history(client)
