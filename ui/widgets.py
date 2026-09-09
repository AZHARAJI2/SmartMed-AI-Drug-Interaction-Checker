from __future__ import annotations

import streamlit as st

from ui.api_client import ApiClient, ScanView

STATUS_BANNER = {"safe": st.success, "caution": st.warning, "danger": st.error}
STATUS_AR = {
    "safe": "آمن — لا يوجد تعارض",
    "caution": "تنبيه — يوجد تعارض يستدعي الحذر",
    "danger": "خطر — يوجد تعارض خطير",
}
SEVERITY_AR = {
    "major": "خطير 🚨",
    "moderate": "متوسط ⚠️",
    "minor": "بسيط",
    "unknown": "غير محدد",
}
FUSION_STATUS_AR = {
    "agree": "اتفاق بين المسارين (قراءة النص + تصنيف الصورة)",
    "ocr_only": "اعتماد على قراءة النص فقط",
    "classifier_only": "اعتماد على تصنيف الصورة فقط",
    "uncertain": "غير مؤكد — يلزم مراجعة صيدلي",
    "no_signal": "لم يتم التعرف على أي دواء",
}
REVIEW_STATUS_AR = {
    "auto": "تلقائي",
    "corrected": "تم تصحيحه",
    "confirmed": "مؤكد",
}


def inject_rtl() -> None:
    """Force the whole Streamlit app to right-to-left (Arabic); drug names stay LTR."""
    st.markdown(
        """
        <style>
            .stApp, .stApp header, section[data-testid="stSidebar"] {
                direction: rtl;
                text-align: right;
            }
            .stMarkdown p, .stMarkdown li, h1, h2, h3, .stCaption, .stAlert {
                text-align: right;
            }
            code, .stCode, .stMetricValue, .stDataFrame {
                direction: ltr;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


COMMON_NAMES_MAP = {
    "acetylsalicylic acid": "أسبرين (Aspirin / Acetylsalicylic Acid)",
    "aspirin": "أسبرين (Aspirin)",
    "acetaminophen": "باراسيتامول / بنادول (Paracetamol / Panadol)",
    "paracetamol": "باراسيتامول (Paracetamol)",
    "panadol": "بنادول (Panadol)",
    "ibuprofen": "بروفين / أدفيل (Ibuprofen / Brufen)",
    "brufen": "بروفين (Brufen)",
    "advil": "أدفيل (Advil)",
    "diclofenac": "فولتارين (Diclofenac / Voltaren)",
    "voltaren": "فولتارين (Voltaren)",
    "amoxicillin": "أموكسيسيلين / أوجمنتين (Amoxicillin / Augmentin)",
    "augmentin": "أوجمنتين (Augmentin)",
    "warfarin": "وارفارين (Warfarin / Coumadin)",
    "clopidogrel": "بلافيكس (Clopidogrel / Plavix)",
    "plavix": "بلافيكس (Plavix)",
    "metformin": "جلوكوفاج (Metformin / Glucophage)",
    "glucophage": "جلوكوفاج (Glucophage)",
    "atorvastatin": "ليبيتور (Atorvastatin / Lipitor)",
    "lipitor": "ليبيتور (Lipitor)",
    "furosemide": "لازيكس (Furosemide / Lasix)",
    "lasix": "لازيكس (Lasix)",
    "omeprazole": "أوميبرازول (Omeprazole)",
    "esomeprazole": "نيكسيوم (Esomeprazole / Nexium)",
    "nexium": "نيكسيوم (Nexium)",
    "ciprofloxacin": "سيبروفلوكساسين (Ciprofloxacin)",
    "cipro": "سيبروفلوكساسين (Cipro)",
    "metronidazole": "فلاجيل (Metronidazole / Flagyl)",
    "flagyl": "فلاجيل (Flagyl)",
    "azithromycin": "زيثروماكس (Azithromycin / Zithromax)",
    "zithromax": "زيثروماكس (Zithromax)",
    "bisoprolol": "كونكور (Bisoprolol / Concor)",
    "concor": "كونكور (Concor)",
}


def render_report(report: dict | None) -> None:
    """Render an InteractionReportOut dict as an Arabic banner + findings table."""
    if report is None:
        st.info("💡 لم يتم استلام تقرير فحص.")
        return
    status = report.get("status", "unknown")
    resolved = report.get("resolved_ingredients") or []
    raw_unmatched = report.get("unmatched_names") or []
    findings = report.get("findings") or []

    # Filter out junk words from unmatched (OCR noise, photographer credits, etc.)
    from config import CONFIG
    unmatched = [name for name in raw_unmatched
                 if not any(j in name.lower() for j in CONFIG.split.junk_keywords)]

    # ── Section 1: Distinction between Recognized and Unrecognized in DB ───────
    col_res, col_unm = st.columns(2)
    with col_res:
        if resolved:
            st.success(
                f"✅ **أدوية تم التعرف عليها في قاعدة البيانات ({len(resolved)}):**\n\n"
                + "  \n".join(f"• `{COMMON_NAMES_MAP.get(name.lower(), name)}`" for name in resolved)
            )
        else:
            st.info("ℹ️ لا توجد أدوية من القائمة مسجلة في قاعدة بيانات DDInter.")

    with col_unm:
        if unmatched:
            st.warning(
                f"⚠️ **أدوية لم يتم التعرف عليها في قاعدة البيانات ({len(unmatched)}):**\n\n"
                + "  \n".join(f"• **{COMMON_NAMES_MAP.get(name.lower(), name)}**" for name in unmatched)
                + "\n\n*(هذه الأدوية غير مسجلة في DDInter ولا يمكن فحص تعارضاتها آليًا)*"
            )
        else:
            st.success("✅ جميع أدويتك تم التعرف عليها بنجاح في قاعدة البيانات!")

    # ── Section 2: Interaction Verdict & Findings ───────────────────────────────
    if findings:
        st.markdown("---")
        banner = STATUS_BANNER.get(status, st.warning)
        banner(f"⚖️ حالة الفحص العامة: **{STATUS_AR.get(status, status)}**")
        st.markdown("#### ⚠️ تفاصيل التفاعلات والتعارضات الدوائية المكتشفة:")

        for idx, f in enumerate(findings):
            d1_name = COMMON_NAMES_MAP.get(f["drug_a"].lower(), f["drug_a"])
            d2_name = COMMON_NAMES_MAP.get(f["drug_b"].lower(), f["drug_b"])
            sev_label = SEVERITY_AR.get(f["severity"].lower(), f["severity"])
            with st.container(border=True):
                col_a, col_vs, col_b = st.columns([5, 1, 5])
                with col_a:
                    st.markdown(f"💊 **الدواء الأول:**")
                    st.markdown(f"##### {d1_name}")
                    st.caption(f"المكون الفعال: `{f.get('ingredient_a', '—')}`")
                with col_vs:
                    st.markdown("<h3 style='text-align: center; color: #dc2626;'>⚠️</h3>", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"💊 **الدواء الثاني:**")
                    st.markdown(f"##### {d2_name}")
                    st.caption(f"المكون الفعال: `{f.get('ingredient_b', '—')}`")

                st.error(f"⚠️ **قيمة الفحص: يوجد تعارض دوائي** (درجة الخطورة: **{sev_label}**)")
                if f.get("description"):
                    st.info(f"📋 **الملاحظة الطبية:** {f['description']}")
                st.caption(f"🏛️ مصدر المعلومة: `{f.get('source', 'DDInter')}`")

        with st.expander("📊 عرض جدول التعارضات المكتشفة"):
            rows = [{
                "الدواء الأول": COMMON_NAMES_MAP.get(f["drug_a"].lower(), f["drug_a"]),
                "الدواء الثاني": COMMON_NAMES_MAP.get(f["drug_b"].lower(), f["drug_b"]),
                "المكون الفعال الأول": f["ingredient_a"],
                "المكون الفعال الثاني": f["ingredient_b"],
                "قيمة التعارض": "يوجد تعارض ⚠️",
                "درجة الخطورة": SEVERITY_AR.get(f["severity"].lower(), f["severity"]),
                "المصدر": f.get("source", ""),
            } for f in findings]
            st.dataframe(rows, hide_index=True)
    elif len(resolved) >= 2:
        st.success("✅ **قيمة الفحص لجميع الأدوية: لا يوجد أي تعارض مسجل** بين الأدوية المعترف بها في قاعدة بيانات DDInter الطبية.")
    elif len(resolved) == 1 and not unmatched:
        st.success("🟢 دواء واحد فقط مسجل — **لا يوجد احتمال لتعارض دوائي** مع دواء مفرد.")
    elif len(resolved) == 1 and unmatched:
        st.info("💡 يوجد دواء واحد فقط معترف به في DDInter — لا يمكن التحقق من تفاعلاته مع الأدوية غير المعترف بها آليًا.")


def find_interaction_for_pair(drug_a: str, drug_b: str, findings: list[dict]) -> dict | None:
    """Find an interaction in findings that matches drug_a and drug_b."""
    da = drug_a.strip().lower()
    db = drug_b.strip().lower()
    for f in findings:
        fa = f.get("drug_a", "").strip().lower()
        fb = f.get("drug_b", "").strip().lower()
        ia = f.get("ingredient_a", "").strip().lower()
        ib = f.get("ingredient_b", "").strip().lower()

        direct = (fa in (da, db) and fb in (da, db)) or (ia in (da, db) and ib in (da, db))
        if direct:
            return f

        sub_match = ((da in fa or fa in da or da in ia or ia in da) and
                     (db in fb or fb in db or db in ib or ib in db))
        if sub_match:
            return f
    return None


def render_pair_card_and_editor(
    drug_a: str,
    drug_b: str,
    finding: dict | None,
    client: ApiClient,
    key_prefix: str,
    on_drug_name_change: callable | None = None,
) -> None:
    """Renders a beautifully structured card showing Drug 1, Drug 2, whether there's an interaction,

    and an interactive modification panel allowing doctor/pharmacist to edit names or toggle interaction status.
    """
    display_a = COMMON_NAMES_MAP.get(drug_a.lower(), drug_a)
    display_b = COMMON_NAMES_MAP.get(drug_b.lower(), drug_b)
    has_inter = (finding is not None)
    severity = finding.get("severity", "Moderate") if finding else "None"
    description = finding.get("description", "") if finding else ""
    source = finding.get("source", "DDInter") if finding else ""

    with st.container(border=True):
        # ── Header: Drug 1 vs Drug 2 ──
        c1, c_mid, c2 = st.columns([5, 1, 5])
        with c1:
            st.markdown("💊 **الدواء الأول:**")
            st.markdown(f"#### {display_a}")
            st.caption(f"الاسم المعتمد: `{drug_a}`")
        with c_mid:
            st.markdown("<h2 style='text-align: center; color: #2563eb; margin-top: 5px;'>⚡</h2>", unsafe_allow_html=True)
        with c2:
            st.markdown("💊 **الدواء الثاني:**")
            st.markdown(f"#### {display_b}")
            st.caption(f"الاسم المعتمد: `{drug_b}`")

        st.markdown("---")

        # ── Verdict: Is there an interaction or not? ──
        c_verdict, c_info = st.columns([3, 2])
        with c_verdict:
            st.markdown("**حالة وقيمة التعارض:**")
            if has_inter:
                sev_label = SEVERITY_AR.get(severity.lower(), severity)
                st.error(f"⚠️ **قيمة الفحص: يوجد تعارض دوائي** (درجة الخطورة: **{sev_label}**)")
                if description:
                    st.info(f"📌 **التفاصيل:** {description}")
            else:
                st.success("✅ **قيمة الفحص: لا يوجد تعارض دوائي** (الاستخدام آمن تماماً)")
                st.caption("لم يتم تسجيل أي تفاعلات سلبية بين هذين الدواءين في قاعدة البيانات.")

        with c_info:
            if has_inter and source:
                st.caption(f"🏛️ مصدر التوثيق: `{source}`")
            elif not has_inter:
                st.caption("🏛️ التوثيق: قاعدة بيانات DDInter الطبية")

        # ── Doctor / Pharmacist Modification Panel ──
        with st.expander("✏️ **تعديل ومراجعة الطبيب / الصيدلاني (تغيير الأسماء أو حالة التعارض)**"):
            st.caption("بإمكانك كطبيب أو صيدلاني تعديل أسماء الأدوية أو تصحيح حكم التعارض (يوجد / لا يوجد تعارض) واعتماده فوراً:")

            c_ed1, c_ed2 = st.columns(2)
            with c_ed1:
                new_a = st.text_input("اسم الدواء الأول:", value=drug_a, key=f"{key_prefix}_ed_a")
            with c_ed2:
                new_b = st.text_input("اسم الدواء الثاني:", value=drug_b, key=f"{key_prefix}_ed_b")

            verdict_opts = ["✅ لا يوجد تعارض (آمن)", "⚠️ يوجد تعارض دوائي"]
            verdict_choice = st.radio(
                "قيمة التعارض المقررة من الطبيب / الصيدلاني:",
                verdict_opts,
                index=1 if has_inter else 0,
                key=f"{key_prefix}_v_choice",
                horizontal=True,
            )
            is_override_inter = (verdict_choice == "⚠️ يوجد تعارض دوائي")

            new_sev = severity if severity in ["Major", "Moderate", "Minor"] else "Moderate"
            new_desc = description
            if is_override_inter:
                c_s, c_d = st.columns([1, 2])
                with c_s:
                    sev_list = ["Major", "Moderate", "Minor"]
                    cur_idx = sev_list.index(new_sev) if new_sev in sev_list else 1
                    chosen_sev = st.selectbox(
                        "درجة خطورة التعارض:",
                        sev_list,
                        index=cur_idx,
                        format_func=lambda s: SEVERITY_AR.get(s.lower(), s),
                        key=f"{key_prefix}_sev_sel",
                    )
                    new_sev = chosen_sev
                with c_d:
                    new_desc = st.text_input(
                        "الملاحظة أو التوصية السريرية:",
                        value=description or "تعارض سريري معتمد بمراجعة الطبيب / الصيدلاني",
                        key=f"{key_prefix}_desc_inp",
                    )

            reviewer_name = st.text_input(
                "اسم أو صفة المراجع:",
                value="طبيب / صيدلاني",
                key=f"{key_prefix}_rev_name"
            )

            if st.button("💾 اعتماد وتطبيق تعديل الطبيب في النظام", type="primary", key=f"{key_prefix}_btn_save"):
                if not new_a.strip() or not new_b.strip():
                    st.warning("يجب تحديد اسم الدواء الأول والثاني.")
                else:
                    try:
                        client.override_interaction(
                            drug_a=new_a.strip(),
                            drug_b=new_b.strip(),
                            has_interaction=is_override_inter,
                            severity=new_sev,
                            description=new_desc,
                            reviewed_by=reviewer_name.strip() or "طبيب / صيدلاني",
                        )
                        st.success("🎉 تم حفظ وتطبيق المراجعة بنجاح في قاعدة البيانات!")
                        if on_drug_name_change and (new_a.strip() != drug_a or new_b.strip() != drug_b):
                            on_drug_name_change(drug_a, drug_b, new_a.strip(), new_b.strip())
                        st.rerun()
                    except Exception as exc:
                        st.error(f"خطأ أثناء حفظ المراجعة: {exc}")



def render_scan_view(view: ScanView, show_ocr: bool = False) -> None:
    """Render a ScanView: verdict, recognition details, findings and the overlay."""
    if not view.accepted:
        st.error(f"تم رفض الصورة بسبب جودتها: {view.rejection_reason}")
        return
    st.subheader("نتيجة التعرف")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("النتيجة النهائية", view.fused_label or "—",
                 f"{view.fused_confidence:.0%}" if view.fused_confidence else None)
    col_b.metric("قراءة النص (OCR)", view.ocr_label or "—")
    col_c.metric("تصنيف الصورة", view.classifier_label or "—")
    fusion_ar = FUSION_STATUS_AR.get(view.fusion_status, view.fusion_status or "—")
    caption = f"حالة الدمج: **{fusion_ar}**"
    if view.matched_drug_name:
        caption += f" · الدواء المطابق: **{view.matched_drug_name}**"
    st.caption(caption)
    overlay = view.annotated_image_rgb
    if overlay is not None:
        st.subheader("الشرح البصري (مواقع النص + النتيجة)")
        st.image(overlay, use_container_width=True)
    if show_ocr and view.ocr_text:
        with st.expander("النص الخام المقروء (OCR)"):
            st.write(view.ocr_text)
    st.subheader("فحص التفاعلات الدوائية")
    render_report(view.report)
