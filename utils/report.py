"""
LeafLab — Report Generator (HTML + PDF via reportlab)
"""
import io
import logging
import datetime

logger = logging.getLogger(__name__)


def generate_html_report(result: dict) -> str:
    """Return an HTML string for the analysis report."""
    f = result.get("features", {})
    s = result.get("species", "Unknown")
    c = result.get("confidence", 0)
    m = result.get("method", "—")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = ""
    for k, v in f.items():
        if isinstance(v, float):
            rows += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        else:
            rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LeafLab Report — {s}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap');
  body {{ font-family: 'Inter', sans-serif; background: #f8f6f1; color: #1a1410; margin: 2rem; }}
  h1 {{ color: #2d6a4f; border-bottom: 2px solid #e0d9cc; padding-bottom: .5rem; }}
  .badge {{ display:inline-block; background:#2d6a4f; color:#fff;
            border-radius:20px; padding:.3rem .9rem; font-size:.85rem; }}
  .conf {{ font-family:'Fira Code',monospace; font-size:1.5rem; color:#2d6a4f; }}
  table {{ border-collapse:collapse; width:100%; background:#fff;
           border-radius:8px; overflow:hidden; }}
  th {{ background:#f2efe8; text-transform:uppercase; font-size:.75rem;
        letter-spacing:.08em; color:#6b5f52; padding:.6rem .8rem; }}
  td {{ padding:.5rem .8rem; border-bottom:1px solid #e0d9cc;
        font-family:'Fira Code',monospace; font-size:.85rem; }}
  tr:last-child td {{ border-bottom:none; }}
  .meta {{ font-size:.8rem; color:#6b5f52; margin:.5rem 0 1.5rem; }}
</style>
</head>
<body>
<h1>🔬 LeafLab Analysis Report</h1>
<p class="meta">Generated: {ts} | Method: {m}</p>
<p>Classification: <strong>{s}</strong></p>
<p>Confidence: <span class="conf">{c*100:.1f}%</span></p>
<h2>Shape Feature Descriptors</h2>
<table>
  <thead><tr><th>Feature</th><th>Value</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</body>
</html>"""
    return html


def generate_pdf_report(result: dict) -> bytes:
    """Return PDF bytes using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER
        import datetime

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        accent = colors.HexColor("#2d6a4f")
        ink    = colors.HexColor("#1a1410")
        ink2   = colors.HexColor("#6b5f52")
        border = colors.HexColor("#e0d9cc")
        bg2    = colors.HexColor("#f2efe8")

        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     textColor=accent, fontSize=22,
                                     fontName="Helvetica-Bold")
        h2_style = ParagraphStyle("h2", parent=styles["Heading2"],
                                  textColor=ink, fontSize=13,
                                  fontName="Helvetica-Bold",
                                  spaceAfter=6)
        body_style = ParagraphStyle("body", parent=styles["Normal"],
                                    textColor=ink2, fontSize=10)

        story = []

        story.append(Paragraph("LeafLab — Morphology Analysis Report", title_style))
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(width="100%", color=border))
        story.append(Spacer(1, 0.4*cm))

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated: {ts}", body_style))
        story.append(Spacer(1, 0.5*cm))

        # Summary table
        s = result.get("species", "Unknown")
        c = result.get("confidence", 0)
        m = result.get("method", "—")
        pt = result.get("process_time", 0)

        summary_data = [
            ["Classification", s],
            ["Confidence", f"{c*100:.1f}%"],
            ["Method", m],
            ["Process Time", f"{pt:.3f}s"],
        ]
        summary_table = Table(summary_data, colWidths=[5*cm, 12*cm])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), bg2),
            ("TEXTCOLOR",  (0, 0), (0, -1), ink2),
            ("TEXTCOLOR",  (1, 0), (1, -1), ink),
            ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, bg2]),
            ("GRID",       (0, 0), (-1, -1), 0.5, border),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.7*cm))

        story.append(Paragraph("Shape Feature Descriptors", h2_style))
        story.append(Spacer(1, 0.2*cm))

        feats = result.get("features", {})
        feat_data = [["Feature", "Value"]]
        for k, v in feats.items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            feat_data.append([k.replace("_", " ").title(), val])

        feat_table = Table(feat_data, colWidths=[8*cm, 9*cm])
        feat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg2]),
            ("GRID",       (0, 0), (-1, -1), 0.5, border),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(feat_table)
        story.append(Spacer(1, 1*cm))

        rec = result.get("recommendation", {})
        if rec:
            story.append(Paragraph("Recommendation", h2_style))
            story.append(Paragraph(f"<b>{rec.get('title','')}</b>", body_style))
            story.append(Paragraph(rec.get("desc", ""), body_style))

        doc.build(story)
        return buffer.getvalue()

    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        # Return minimal valid PDF fallback
        fallback = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
        )
        return fallback
