"""صفحة الطبيب / الصيدلاني — سلة فحص تفاعلية، فحص العبوات بالكاميرا/الملف، وتدريب النظام بسهولة."""
from __future__ import annotations

import streamlit as st

from config import CONFIG
from ui.api_client import ApiClient, ApiError
from ui.pdf_report import generate_patient_pdf_report
from ui.widgets import COMMON_NAMES_MAP, REVIEW_STATUS_AR, render_report, render_scan_view


def _client() -> ApiClient:
    return ApiClient(st.session_state.api_base_url)


def _init_session() -> None:
    if "doctor_basket" not in st.session_state:
        st.session_state.doctor_basket = []
    if "last_scanned_id" not in st.session_state:
        st.session_state.last_scanned_id = None


def _clean_drug_name(name: str) -> str:
    """Strip junk words from drug candidate name."""
    cleaned = name.strip()
    for junk in CONFIG.split.junk_keywords:
        if junk in cleaned.lower():
            return ""
    return cleaned


def _render_medication_basket() -> None:
    """عرض سلة الأدوية الحالية مع إمكانية حذف أي علاج وفحص التعارض التلقائي."""
    st.markdown("---")
    st.subheader("📋 سلة الأدوية الحالية للفحص")

    basket: list[str] = st.session_state.doctor_basket

    if not basket:
        st.info("💡 سلة الفحص فارغة حاليًا. يمكنك إضافة أدوية بالاسم من تبويب **(⚡ فحص فوري بالاسم)** أو تصوير العبوة من تبويب **(📷 فحص صورة عبوة)**.")
        return

    st.write(f"الأدوية المسجلة حاليًا في جلسة الفحص (**{len(basket)}** أدوية):")

    # Display each medication with a delete button
    for idx, drug in enumerate(list(basket)):
        col_info, col_del = st.columns([4, 1])
        display_name = COMMON_NAMES_MAP.get(drug.lower(), drug)
        with col_info:
            st.markdown(f"💊 **{display_name}** (`{drug}`)")
        with col_del:
            if st.button("❌ حذف", key=f"doc_basket_del_{idx}_{drug}"):
                st.session_state.doctor_basket.pop(idx)
                st.rerun()

    top_actions = st.columns([2, 2])
    with top_actions[0]:
        if st.button("🗑️ تفريغ كل الأدوية من السلة", type="secondary", key="doc_basket_clear_all"):
            st.session_state.doctor_basket = []
            st.rerun()

    # Run interaction check
    st.markdown("### 🔬 نتيجة فحص التعارض الدوائي للسلة")
    if len(basket) == 1:
        single_name = basket[0]
        friendly_single = COMMON_NAMES_MAP.get(single_name.lower(), single_name)
        st.success(f"🟢 دواء واحد فقط مسجل في السلة (**{friendly_single}**) — لا يمكن حدوث تعارض دوائي مع دواء مفرد.\n\n"
                   f"👉 **الخطوة التالية:** أضف أو صوّر دواءً آخر وسيقوم النظام فوراً بمقارنته مع **{friendly_single}** واكتشاف أي تفاعلات سلبية بينهما.")
    else:
        try:
            report = _client().doctor_check(basket)
            render_report(report)

            # PDF Download option for doctor/pharmacist
            med_list = [{"drug_name": d, "drug_id": i + 1} for i, d in enumerate(basket)]
            pdf_bytes = generate_patient_pdf_report(
                patient_email="Doctor_Pharmacist_Session",
                medications=med_list,
                interaction_report=report,
            )
            st.download_button(
                label="📥 تحميل تقرير التعارض الطبي (PDF)",
                data=pdf_bytes,
                file_name="Medical_Interaction_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ApiError as exc:
            st.error(f"خطأ في فحص التعارض: {exc}")


def _render_instant_check() -> None:
    st.write("أضف أي دواء إلى سلة الفحص بالاسم العلمي أو التجاري — يُقبل أي اسم حتى لو لم يكن في قاعدة البيانات. "
             "التعارض يُفحص فقط للأدوية الموجودة في DDInter، والباقي يُبيّن أنه غير موجود في القاعدة.")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "اكتب اسم الدواء مباشرة (مثال: Aspirin, Ibuprofen, Panadol, كلوبيدوجريل):",
            key="quick_search_input",
            placeholder="أكتب أي اسم دواء تجاري أو علمي"
        )
    with col2:
        st.write("")
        st.write("")
        add_quick = st.button("➕ إضافة للسلة", key="btn_quick_add", type="primary")

    # Show DB suggestions as optional picker — not required
    if query:
        candidates = _client().search_drugs(query, limit=8)
        if candidates:
            cand_names = [c["trade_name"] for c in candidates]
            chosen = st.selectbox(
                "💡 وُجد هذا الدواء في القاعدة — يمكنك اختياره للحصول على فحص تعارض أدق:",
                cand_names, key="cand_select"
            )
            if st.button("➕ إضافة الدواء المطابق من القاعدة", key="btn_add_cand"):
                if chosen not in st.session_state.doctor_basket:
                    st.session_state.doctor_basket.append(chosen)
                    st.success(f"تمت إضافة **{chosen}** إلى السلة!")
                    st.rerun()
                else:
                    st.warning("هذا الدواء موجود بالفعل في السلة.")

    # Direct add — any name accepted
    if add_quick and query.strip():
        clean_q = query.strip()  # Accept as-is (only filter true junk)
        for junk in ["paolo", "monti", "servizio", "fotografico", "beic", "wikimedia"]:
            if junk in clean_q.lower():
                clean_q = ""
                break
        if clean_q:
            if clean_q not in st.session_state.doctor_basket:
                st.session_state.doctor_basket.append(clean_q)
                st.success(f"✅ تمت إضافة **{clean_q}** إلى السلة!")
                st.rerun()
            else:
                st.warning("هذا الدواء موجود بالفعل في السلة.")
        else:
            st.warning("الاسم المُدخل يبدو غير صحيح (كلمات غير دوائية).")

    with st.expander("📝 إضافة عدة أدوية دفعة واحدة"):
        raw_list = st.text_area("أدخل كل دواء في سطر مستقل أو مفصولاً بفواصل", key="bulk_names",
                                placeholder="Aspirin\nWarfarin\nبنادول\nMetformin")
        if st.button("➕ إضافة جميع هذه الأدوية إلى السلة", key="btn_bulk_add"):
            names = [part.strip() for line in raw_list.splitlines()
                     for part in line.split(",") if part.strip()]
            added_count = 0
            for n in names:
                clean_n = _clean_drug_name(n)
                if not clean_n:
                    clean_n = n.strip()  # Fallback: keep original if _clean_drug_name returns empty
                if clean_n and clean_n not in st.session_state.doctor_basket:
                    st.session_state.doctor_basket.append(clean_n)
                    added_count += 1
            if added_count > 0:
                st.success(f"تمت إضافة {added_count} أدوية بنجاح إلى السلة!")
                st.rerun()



def _render_scan() -> None:
    st.write("صوّر علبة أو شريط الدواء مباشرة أو ارفع صورة؛ سيتعرف النظام على الدواء ويضيفه تلقائياً إلى سلة الفحص لمقارنته مع أي أدوية سابقة.")

    input_mode = st.radio(
        "طريقة إدخال الصورة",
        ["📸 تصوير مباشر بالكاميرا", "📁 رفع صورة من الجهاز"],
        horizontal=True,
        key="doctor_input_mode",
    )
    image_bytes = None
    image_name = "package.jpg"
    if input_mode == "📸 تصوير مباشر بالكاميرا":
        cam = st.camera_input("وجّه الكاميرا نحو علبة أو شريط الدواء بوضوح", key="doctor_cam")
        if cam is not None:
            image_bytes = cam.getvalue()
            image_name = "doctor_cam.jpg"
    else:
        upload = st.file_uploader("صورة عبوة الدواء", type=["jpg", "jpeg", "png"], key="doctor_scan")
        if upload is not None:
            image_bytes = upload.getvalue()
            image_name = upload.name

    if image_bytes is not None and st.button("🔍 فحص الصورة الآن", type="primary", key="btn_doctor_scan"):
        with st.spinner("جارٍ الفحص (فحص الجودة ← تحسين الصورة ← قراءة النص OCR + تصنيف الصورة ← الدمج)..."):
            try:
                view = _client().doctor_scan(image_bytes, image_name)
            except ApiError as exc:
                st.error(str(exc))
                return

        render_scan_view(view, show_ocr=True)

        if view.accepted:
            st.session_state.review_scan_id = view.scan_id
            st.session_state.review_ocr_text = view.ocr_text

            # Extract detected drug name
            candidate = view.matched_drug_name or view.fused_label or view.ocr_label
            detected = _clean_drug_name(candidate)

            if detected:
                # Add to basket if new scan
                if view.scan_id != st.session_state.last_scanned_id:
                    st.session_state.last_scanned_id = view.scan_id
                    if detected not in st.session_state.doctor_basket:
                        st.session_state.doctor_basket.append(detected)
                        st.success(f"✅ تم التعرف على الدواء: **{detected}** وإضافته تلقائياً إلى سلة الفحص!")

            # Inline correction widget for pharmacists
            with st.expander("🛠️ هل قراءة النظام لهذا الفحص غير دقيقة؟ يمكنك تصحيحها وتدريب النظام فوراً من هنا"):
                st.caption("اختر الدواء الصحيح من قاعدة البيانات وسيتعلم النظام الربط بين النص المقروء والدواء تلقائياً لعدم تكرار الخطأ:")
                c_query = st.text_input("ابحث عن الدواء الصحيح", key=f"inline_corr_q_{view.scan_id}")
                candidates = _client().search_drugs(c_query) if c_query else []
                options = [c["trade_name"] for c in candidates]
                choice = st.selectbox("الدواء المعتمد الصحيح", options, key=f"inline_corr_choice_{view.scan_id}") if options else ""
                if st.button("💾 اعتماد وتدريب التصحيح في النظام", key=f"btn_inline_corr_{view.scan_id}"):
                    if choice and view.scan_id is not None:
                        try:
                            _client().doctor_review(view.scan_id, view.ocr_text or detected, choice, "صيدلاني")
                            # Update basket
                            if detected in st.session_state.doctor_basket:
                                idx = st.session_state.doctor_basket.index(detected)
                                st.session_state.doctor_basket[idx] = choice
                            elif choice not in st.session_state.doctor_basket:
                                st.session_state.doctor_basket.append(choice)
                            st.success(f"تم تصحيح وتدريب النظام بنجاح: تم اعتماد **{choice}**!")
                            st.rerun()
                        except ApiError as exc:
                            st.error(str(exc))


def _render_review() -> None:
    st.subheader("✍️ تدريب الذكاء الاصطناعي ومراجعة الفحوصات")
    st.info(
        "💡 **كيف يعمل هذا القسم؟**\n\n"
        "هذا القسم مخصص للصيادلة والأطباء لتعليم النظام وتطوير دقته.\n"
        "عندما يتعرف النظام على صورة أو نص دواء بشكل خاطئ أو غير مكتمل، اختر الفحص من القائمة أدناه وحدد الدواء الصحيح، "
        "وسيقوم النظام بحفظ هذا التصحيح في القاموس الذكي ليتعرف عليه مستقبلاً بدقة 100% تلقائيًا."
    )

    try:
        scans = _client().recent_scans(limit=30)
    except ApiError as exc:
        st.error(f"تعذر جلب الفحوصات الأخيرة: {exc}")
        scans = []

    if not scans:
        st.warning("لا توجد فحوصات مسجلة بعد. قم بتصوير عبوة دواء في تبويب (📷 فحص صورة عبوة) لتظهر هنا.")
        return

    # Build human-readable options so the pharmacist never has to guess a Scan ID!
    scan_options = {}
    for s in scans:
        s_id = s["scan_id"]
        ocr_preview = (s["ocr_raw_text"] or "").replace("\n", " ").strip()
        if len(ocr_preview) > 35:
            ocr_preview = ocr_preview[:32] + "..."
        ocr_preview = ocr_preview or "بدون نص"
        res = s.get("classifier_prediction") or s.get("matched_drug_id") or "غير محدد"
        status_ar = REVIEW_STATUS_AR.get(s["review_status"], s["review_status"])
        label = f"فحص #{s_id} | النص: [{ocr_preview}] | النتيجة: [{res}] | الحالة: [{status_ar}]"
        scan_options[label] = s

    selected_label = st.selectbox("📌 اختر الفحص الذي تريد مراجعته أو تصحيحه:", list(scan_options.keys()))
    selected_scan = scan_options[selected_label]

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**رقم الفحص (Scan ID):** `{selected_scan['scan_id']}`")
        st.markdown(f"**حالة المراجعة الحالية:** `{REVIEW_STATUS_AR.get(selected_scan['review_status'], selected_scan['review_status'])}`")
        if selected_scan.get("reviewed_by"):
            st.markdown(f"**تمت المراجعة بواسطة:** `{selected_scan['reviewed_by']}`")
    with col2:
        st.markdown(f"**توقع النموذج:** `{selected_scan.get('classifier_prediction') or '—'}`")
        st.markdown(f"**نسبة الثقة:** `{selected_scan.get('fused_confidence', 0):.0%}`")

    raw_text = st.text_area(
        "النص الخام الذي قرأه الـ OCR من العبوة (يمكنك تعديله لتدريب الكلمات الدقيقة):",
        value=selected_scan.get("ocr_raw_text", ""),
        height=70,
        key=f"rev_ocr_{selected_scan['scan_id']}",
    )

    st.markdown("#### حدد الدواء الصحيح من قاعدة البيانات:")
    query = st.text_input("ابحث باسم الدواء التجاري أو العلمي (مثال: Panadol أو Aspirin)", key=f"rev_q_{selected_scan['scan_id']}")
    candidates = _client().search_drugs(query) if query else []
    options = [c["trade_name"] for c in candidates]

    choice = ""
    if options:
        choice = st.selectbox("اختر الدواء الصحيح المعتمد:", options, key=f"rev_choice_{selected_scan['scan_id']}")
    elif query:
        st.warning("لم يتم العثور على دواء مطابق — تحقق من صحة كتابة الاسم الإنجليزي.")

    reviewed_by = st.text_input("اسم أو صفة المراجع", value="صيدلاني", key=f"rev_by_{selected_scan['scan_id']}")

    if st.button("💾 اعتماد وتدريب التصحيح في النظام", type="primary", key=f"btn_save_rev_{selected_scan['scan_id']}"):
        if not raw_text.strip() or not choice:
            st.warning("يجب توفير كل من النص المقروء والدواء الصحيح.")
            return
        try:
            report = _client().doctor_review(int(selected_scan["scan_id"]), raw_text, choice, reviewed_by)
            st.success(f"🎉 تم تعلم التصحيح بنجاح! تم ربط النص “{raw_text.strip()}” بالدواء المعتمد **{choice}**. سيتم التعرف عليه تلقائيًا في جميع الفحوصات القادمة.")
            render_report(report)
        except ApiError as exc:
            st.error(str(exc))


def render_doctor_page() -> None:
    _init_session()
    st.subheader("محطة الطبيب / الصيدلاني (دخول فوري بدون تسجيل)")
    tabs = st.tabs(["⚡ فحص فوري بالاسم", "📷 فحص صورة عبوة", "✍️ مراجعة وتصحيح النظام"])
    with tabs[0]:
        _render_instant_check()
    with tabs[1]:
        _render_scan()
    with tabs[2]:
        _render_review()

    _render_medication_basket()
