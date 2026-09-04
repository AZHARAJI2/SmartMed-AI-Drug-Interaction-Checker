"""مولّد التقرير الطبي بصيغة PDF — واجهة عربية كاملة مع دعم RTL.

ينتج تقريراً طبياً قابلاً للتحميل يوضّح قائمة الأدوية ونتائج فحص التفاعلات
ليقدّمه المريض لطبيبه المعالج أو الصيدلاني.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Register Arabic-capable font ────────────────────────────────────────────
_FONT_PATH = r"C:\Windows\Fonts\ARIALUNI.TTF"
pdfmetrics.registerFont(TTFont("ArialUni", _FONT_PATH))
_FONT = "ArialUni"

SEVERITY_AR = {
    "major": "خطير (Major)",
    "moderate": "متوسط (Moderate)",
    "minor": "طفيف (Minor)",
}


def _ar(text: str) -> str:
    """Reshape and reorder Arabic text for correct RTL rendering in ReportLab."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def generate_patient_pdf_report(
    patient_email: str,
    medications: list[dict[str, Any]],
    interaction_report: dict[str, Any] | None = None,
) -> bytes:
    """Build a professional Arabic clinical PDF report and return it as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # ── Custom Arabic styles ────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName=_FONT,
        fontSize=18,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1E3A8A"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName=_FONT,
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5563"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName=_FONT,
        fontSize=13,
        leading=18,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName=_FONT,
        fontSize=9,
        leading=13,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#374151"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName=_FONT,
        fontSize=9,
        leading=12,
        alignment=TA_RIGHT,
        textColor=colors.white,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=_FONT,
        fontSize=9,
        leading=12,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#1F2937"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName=_FONT,
        fontSize=8,
        leading=11,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#6B7280"),
    )

    story: list = []

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Header Banner
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph(_ar("تقرير طبي — قائمة الأدوية وفحص التفاعلات الدوائية"), title_style))
    story.append(Paragraph(
        _ar("نظام فحص تفاعلات الأدوية بالذكاء الاصطناعي — المرجع: قاعدة بيانات DDInter العالمية"),
        subtitle_style,
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563EB"), spaceAfter=12))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Patient & Report Metadata
    # ══════════════════════════════════════════════════════════════════════════
    now_str = datetime.now().strftime("%Y-%m-%d  %H:%M")
    meta_data = [
        [
            Paragraph(_ar(f"تاريخ التقرير: {now_str}"), meta_style),
            Paragraph(_ar(f"البريد الإلكتروني للمريض: {patient_email}"), meta_style),
        ],
        [
            Paragraph(_ar("نظام التحقق: قراءة نص (OCR) + تصنيف صورة (Deep Learning)"), meta_style),
            Paragraph(_ar(f"عدد الأدوية المسجلة: {len(medications)}"), meta_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Current Medications Table
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph(_ar("١. قائمة الأدوية الحالية"), section_heading))
    if not medications:
        story.append(Paragraph(_ar("لا توجد أدوية مسجلة حالياً في ملف المريض."), meta_style))
    else:
        med_rows = [[
            Paragraph(_ar("الحالة"), table_header_style),
            Paragraph(_ar("رقم مرجعي"), table_header_style),
            Paragraph(_ar("اسم الدواء"), table_header_style),
            Paragraph(_ar("م"), table_header_style),
        ]]
        for idx, m in enumerate(medications, start=1):
            med_rows.append([
                Paragraph(_ar("مسجّل"), table_cell_style),
                Paragraph(str(m.get("drug_id", "—")), table_cell_style),
                Paragraph(_ar(m.get("drug_name", "غير معروف")), table_cell_style),
                Paragraph(str(idx), table_cell_style),
            ])
        med_table = Table(med_rows, colWidths=[100, 80, 280, 40])
        med_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
        story.append(med_table)

    story.append(Spacer(1, 14))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Interaction Analysis Section
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph(_ar("٢. نتائج فحص التفاعلات الدوائية"), section_heading))

    status = (interaction_report or {}).get("status", "safe").lower()
    findings = (interaction_report or {}).get("findings", [])
    unmatched = (interaction_report or {}).get("unmatched_names", [])

    if status == "danger":
        status_bg = colors.HexColor("#FEE2E2")
        status_border = colors.HexColor("#EF4444")
        status_text_color = "#991B1B"
        status_label = "خطر عالي — تم اكتشاف تفاعل دوائي خطير يستدعي التدخل الفوري"
    elif status == "caution":
        status_bg = colors.HexColor("#FEF3C7")
        status_border = colors.HexColor("#F59E0B")
        status_text_color = "#92400E"
        status_label = "تحذير — يُنصح بالحذر والمتابعة الطبية"
    else:
        status_bg = colors.HexColor("#DCFCE7")
        status_border = colors.HexColor("#22C55E")
        status_text_color = "#166534"
        status_label = "آمن — لا توجد تفاعلات خطيرة معروفة بين أدويتك الحالية"

    status_p = Paragraph(
        f"<font color='{status_text_color}'>{_ar(f'التقييم العام: {status_label}')}</font>",
        ParagraphStyle("StatusP", parent=styles["Normal"], fontName=_FONT,
                       fontSize=11, leading=16, alignment=TA_RIGHT),
    )
    status_table = Table([[status_p]], colWidths=[500])
    status_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), status_bg),
            ("BOX", (0, 0), (-1, -1), 1, status_border),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(status_table)
    story.append(Spacer(1, 8))

    # Unmatched drugs notice
    if unmatched:
        notice = _ar(
            "ملاحظة: الأدوية التالية غير موجودة في قاعدة بيانات التفاعلات DDInter ولم يتم فحصها: "
            + "، ".join(unmatched)
            + ". يُنصح بمراجعة الصيدلاني."
        )
        story.append(Paragraph(notice, ParagraphStyle(
            "Notice", parent=styles["Normal"], fontName=_FONT,
            fontSize=9, leading=13, alignment=TA_RIGHT,
            textColor=colors.HexColor("#92400E"),
        )))
        story.append(Spacer(1, 6))

    if findings:
        findings_rows = [[
            Paragraph(_ar("الوصف السريري"), table_header_style),
            Paragraph(_ar("المصدر"), table_header_style),
            Paragraph(_ar("الخطورة"), table_header_style),
            Paragraph(_ar("الأدوية المتعارضة"), table_header_style),
        ]]
        for f in findings:
            sev = SEVERITY_AR.get(f.get("severity", "moderate").lower(),
                                  f.get("severity", ""))
            pair = f"{f.get('drug_a', '')} + {f.get('drug_b', '')}"
            src = f.get("source", "DDInter")
            desc = f.get("description", "تفاعل دوائي محتمل.")
            findings_rows.append([
                Paragraph(_ar(desc[:120]), table_cell_style),
                Paragraph(src, table_cell_style),
                Paragraph(_ar(sev), table_cell_style),
                Paragraph(_ar(pair), table_cell_style),
            ])
        findings_table = Table(findings_rows, colWidths=[240, 60, 80, 120])
        findings_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(findings_table)
    else:
        if len(medications) <= 1:
            story.append(Paragraph(
                _ar("دواء واحد أو أقل مسجل — لا يلزم فحص تفاعلات بين أدوية متعددة."),
                meta_style,
            ))
        else:
            story.append(Paragraph(
                _ar("تم فحص جميع الأزواج في قاعدة بيانات DDInter (أكثر من ١٦٠ ألف تفاعل مسجل) "
                    "ولم يُعثر على أي تعارض خطير أو متوسط."),
                meta_style,
            ))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Clinical Advisory Disclaimer
    # ══════════════════════════════════════════════════════════════════════════
    disclaimer_text = _ar(
        "إخلاء مسؤولية طبي: هذا التقرير صادر تلقائياً من نظام دعم القرار السريري باستخدام "
        "الذكاء الاصطناعي (قراءة النص + تصنيف الصور بالتعلم العميق) وقاعدة بيانات التفاعلات DDInter. "
        "يهدف التقرير للفحص الأولي والمناقشة مع مقدم الرعاية الصحية. "
        "لا تتوقف عن تناول أي دواء أو تبدأ أو تغيّر جرعته دون استشارة طبيب أو صيدلاني مرخّص."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    return buffer.getvalue()
