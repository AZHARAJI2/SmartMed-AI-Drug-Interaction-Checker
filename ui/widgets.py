"""Shared Streamlit rendering helpers for both pages (Arabic UI)."""
from __future__ import annotations

import streamlit as st

from ui.api_client import ScanView

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
        st.info("💡 تم تحديد الدواء بنجاح. لبدء فحص التعارض الدوائي، يلزم توفر دوائين أو أكثر لمقارنتهما معًا.")
        return
    status = report.get("status", "unknown")
    resolved = report.get("resolved_ingredients") or []
    raw_unmatched = report.get("unmatched_names") or []
    findings = report.get("findings") or []

    # Filter out junk words from unmatched (OCR noise, photographer credits, etc.)
    from config import CONFIG
    unmatched = [name for name in raw_unmatched
                 if not any(j in name.lower() for j in CONFIG.split.junk_keywords)]

    # ── CASE 1: single drug, nothing to compare ────────────────────────────────
    if not findings and (len(resolved) + len(unmatched)) <= 1:
        st.success("🟢 دواء واحد فقط مسجل — لا يمكن حدوث تعارض دوائي مع دواء مفرد.\n\n"
                   "👉 أضف أو صوّر دواءً ثانياً لبدء فحص التعارض بينهما تلقائياً.")
        if resolved:
            st.markdown("المكون الفعال للدواء: "
                        + ", ".join(f"`{COMMON_NAMES_MAP.get(name.lower(), name)}`" for name in resolved))
        if unmatched:
            st.info(f"ℹ️ الدواء **{unmatched[0]}** غير موجود في قاعدة بيانات التفاعلات DDInter، "
                    "لذلك لا يمكن التحقق من تعارضاته — يُنصح بمراجعة الصيدلاني.")
        return

    # ── CASE 2: multiple drugs — show interaction result ──────────────────────
    banner = STATUS_BANNER.get(status, st.info)
    banner(f"النتيجة: **{STATUS_AR.get(status, status)}**")

    if resolved:
        st.markdown("✅ **أدوية تم التعرف عليها في قاعدة البيانات:**\n"
                    + "  \n".join(f"• `{COMMON_NAMES_MAP.get(name.lower(), name)}`" for name in resolved))

    # Drugs not in the reference DB — shown informatively, NOT as a scary error
    if unmatched:
        st.info(
            "ℹ️ **أدوية غير موجودة في قاعدة بيانات التفاعلات (DDInter):**\n"
            + "  \n".join(f"• **{name}**" for name in unmatched)
            + "\n\nلا يمكن فحص تعارض هذه الأدوية تلقائياً. يُنصح بمراجعة الصيدلاني أو الطبيب."
        )

    if findings:
        st.markdown("---")
        st.markdown("#### ⚠️ التفاعلات الدوائية المكتشفة:")
        rows = [{
            "العلاج الأول": COMMON_NAMES_MAP.get(f["drug_a"].lower(), f["drug_a"]),
            "العلاج الثاني": COMMON_NAMES_MAP.get(f["drug_b"].lower(), f["drug_b"]),
            "المكون الفعال الأول": f["ingredient_a"],
            "المكون الفعال الثاني": f["ingredient_b"],
            "درجة التعارض": SEVERITY_AR.get(f["severity"].lower(), f["severity"]),
            "المصدر": f.get("source", ""),
        } for f in findings]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    elif status == "safe" and resolved:
        st.success("✅ لا يوجد أي تعارض مسجل بين هذه الأدوية في قاعدة بيانات DDInter.")



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
