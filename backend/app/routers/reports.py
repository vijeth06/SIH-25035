"""
Report generation endpoints (Excel exports) for officials.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import pandas as pd

from backend.app.core.database import get_db
from backend.app.core.security import require_staff_or_admin
from backend.app.models.user import User
from backend.app.models.comment import Comment
from backend.app.models.analysis import AnalysisResult, SentimentLabel
from backend.app.services.visualization_service import VisualizationService

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

viz = VisualizationService()


def _fetch_analysis_rows(db: Session, consultation_id: str):
    q = (
        db.query(
            Comment.id,
            Comment.original_text,
            Comment.law_section,
            Comment.submitted_at,
            AnalysisResult.sentiment_label,
            AnalysisResult.sentiment_confidence,
            AnalysisResult.summary,
        )
        .join(AnalysisResult, AnalysisResult.comment_id == Comment.id)
        .filter(Comment.consultation_id == consultation_id)
    )
    rows = q.all()
    return rows


@router.get("/excel")
async def export_excel(
    consultation_id: str = Query(..., description="Consultation/draft ID"),
    current_user: User = Depends(require_staff_or_admin),
    db: Session = Depends(get_db),
):
    """
    Export an Excel report with:
    - Summary sheet: sentiment distribution
    - Comments sheet: comments and analysis
    - Keywords sheet: top keywords
    """
    try:
        rows = _fetch_analysis_rows(db, consultation_id)
        if not rows:
            raise HTTPException(status_code=404, detail="No analyzed data found for this consultation")

        # Build DataFrame of comments
        df = pd.DataFrame(rows, columns=[
            "comment_id", "text", "law_section", "submitted_at", "sentiment", "confidence", "summary"
        ])

        # Sentiment distribution
        sentiment_counts = df["sentiment"].value_counts().reindex(
            [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL], fill_value=0
        )
        dist_df = pd.DataFrame({
            "sentiment": ["positive", "negative", "neutral"],
            "count": [int(sentiment_counts.get(lbl, 0)) for lbl in [
                SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL
            ]]
        })

        # Top keywords
        texts = df["summary"].fillna("").tolist()
        # fallback to original text if summary empty
        if not any(texts):
            texts = df["text"].fillna("").tolist()
        tokens = await viz.prepare_tokens(texts)
        freq = viz.compute_frequencies(tokens, max_words=200)
        kw_df = pd.DataFrame([{"keyword": k, "count": v} for k, v in freq.items()])

        # Write to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            dist_df.to_excel(writer, sheet_name="Summary", index=False)
            df.to_excel(writer, sheet_name="Comments", index=False)
            kw_df.to_excel(writer, sheet_name="Keywords", index=False)
        output.seek(0)

        filename = f"consultation_{consultation_id}_report.xlsx"
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export report: {str(e)}")


@router.get("/pdf")
async def export_pdf(
    consultation_id: str = Query(..., description="Consultation/draft ID"),
    current_user: User = Depends(require_staff_or_admin),
    db: Session = Depends(get_db),
):
    """
    Export a PDF report with sentiment distribution, top keywords, and sample comments.
    """
    try:
        rows = _fetch_analysis_rows(db, consultation_id)
        if not rows:
            raise HTTPException(status_code=404, detail="No analyzed data found for this consultation")

        # Prepare data
        df = pd.DataFrame(rows, columns=[
            "comment_id", "text", "law_section", "submitted_at", "sentiment", "confidence", "summary"
        ])

        # Sentiment counts
        sentiment_counts = df["sentiment"].value_counts()
        pos = int(sentiment_counts.get(SentimentLabel.POSITIVE, 0))
        neg = int(sentiment_counts.get(SentimentLabel.NEGATIVE, 0))
        neu = int(sentiment_counts.get(SentimentLabel.NEUTRAL, 0))

        # Keywords from summaries/texts
        texts = df["summary"].fillna("").tolist()
        if not any(texts):
            texts = df["text"].fillna("").tolist()
        tokens = await viz.prepare_tokens(texts)
        freq = viz.compute_frequencies(tokens, max_words=50)
        keywords_table_data = [["Keyword", "Count"]] + [[k, v] for k, v in freq.items()]

        # Build PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(f"Consultation Report: {consultation_id}", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Sentiment Distribution", styles["Heading2"]))
        sent_table = Table([
            ["Positive", "Negative", "Neutral", "Total"],
            [str(pos), str(neg), str(neu), str(pos + neg + neu)],
        ])
        sent_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(sent_table)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Top Keywords", styles["Heading2"]))
        kw_table = Table(keywords_table_data)
        kw_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(kw_table)
        story.append(Spacer(1, 12))

        # Sample comments table (first 10)
        story.append(Paragraph("Sample Comments", styles["Heading2"]))
        sample_df = df.head(10)[["comment_id", "law_section", "sentiment", "confidence", "summary"]]
        table_data = [["ID", "Section", "Sentiment", "Confidence", "Summary"]] + sample_df.values.tolist()
        sample_table = Table(table_data, repeatRows=1)
        sample_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(sample_table)

        doc.build(story)
        buffer.seek(0)
        filename = f"consultation_{consultation_id}_report.pdf"
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export PDF: {str(e)}")