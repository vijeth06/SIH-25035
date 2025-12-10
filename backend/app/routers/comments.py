"""
Comment submission and retrieval endpoints for stakeholder and analyst flows.
"""
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import get_current_active_user, get_optional_current_user
from backend.app.models.user import User, UserRole
from backend.app.models.comment import Comment

router = APIRouter(prefix="/api/v1/comments", tags=["comments"])

# Simple in-memory rate limiter (per-IP)
_RATE_LIMIT_WINDOW_SEC = 60
_RATE_LIMIT_MAX_REQUESTS = 10
_RATE_LIMIT_BUCKET: Dict[str, List[datetime]] = {}


def _rate_limit_check(client_key: str) -> None:
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=_RATE_LIMIT_WINDOW_SEC)
    timestamps = _RATE_LIMIT_BUCKET.get(client_key, [])
    # Drop old
    timestamps = [t for t in timestamps if t > window_start]
    if len(timestamps) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    timestamps.append(now)
    _RATE_LIMIT_BUCKET[client_key] = timestamps


def _is_spammy(text: str) -> Optional[str]:
    t = text.strip()
    if len(t) < 30:
        return "Comment text too short (min 30 characters)"
    # Excessive links
    import re
    links = len(re.findall(r"https?://\S+", t))
    if links > 2:
        return "Too many links detected"
    # Repeated characters
    if re.search(r"(.)\1{7,}", t):
        return "Excessive repeated characters detected"
    # Very low unique word ratio
    words = [w.lower() for w in re.findall(r"[A-Za-z\u0900-\u097F]{2,}", t)]
    if words:
        unique_ratio = len(set(words)) / max(1, len(words))
        if unique_ratio < 0.2 and len(words) > 20:
            return "Low content diversity detected"
    return None

class CommentCreate(BaseModel):
    draft_id: str = Field(..., description="Draft legislation/consultation ID")
    text: str = Field(..., min_length=10, max_length=20000, description="Comment text")
    section: Optional[str] = Field(None, description="Section/Provision reference")
    stakeholder_type: Optional[str] = None
    stakeholder_category: Optional[str] = None
    location: Optional[str] = None
    submitted_at: Optional[datetime] = None


class ModerationAction(str, Enum):
    FLAG = "flag"
    APPROVE = "approve"
    OVERRIDE = "override"

class ModerationRequest(BaseModel):
    comment_id: int
    action: ModerationAction
    new_label: Optional[str] = None
    reason: Optional[str] = None

class ModerationResponse(BaseModel):
    success: bool
    message: str

class LanguageStatsResponse(BaseModel):
    language_counts: Dict[str, int]
    underrepresented: List[str]


class CommentResponse(BaseModel):
    id: int
    consultation_id: Optional[str]
    law_section: Optional[str]
    original_text: str
    created_at: Optional[datetime]

@router.post("/moderate", response_model=ModerationResponse)
async def moderate_comment(
    request: ModerationRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Moderator endpoint to flag, approve, or override comment sentiment/label.
    """
    if current_user.role not in [UserRole.ADMIN, UserRole.STAFF, UserRole.MODERATOR]:
        raise HTTPException(status_code=403, detail="Insufficient privileges for moderation")
    comment = db.query(Comment).filter(Comment.id == request.comment_id).first()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    # Example: update status/label based on action
    if request.action == ModerationAction.FLAG:
        comment.flagged = True
        comment.flag_reason = request.reason
        db.commit()
        return ModerationResponse(success=True, message="Comment flagged for review.")
    elif request.action == ModerationAction.APPROVE:
        comment.flagged = False
        comment.flag_reason = None
        db.commit()
        return ModerationResponse(success=True, message="Comment approved.")
    elif request.action == ModerationAction.OVERRIDE:
        if request.new_label:
            comment.sentiment_label = request.new_label
            db.commit()
            return ModerationResponse(success=True, message=f"Comment label overridden to {request.new_label}.")
        else:
            raise HTTPException(status_code=400, detail="New label required for override.")
    else:
        raise HTTPException(status_code=400, detail="Invalid moderation action.")

@router.get("/language-stats", response_model=LanguageStatsResponse)
async def get_language_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Show comment volume by language and alert if underrepresented.
    """
    # Example: aggregate by detected_language field
    from sqlalchemy import func
    q = db.query(getattr(Comment, "detected_language"), func.count(Comment.id)).group_by(getattr(Comment, "detected_language"))
    results = q.all()
    language_counts = {lang: count for lang, count in results if lang}
    # Alert if any major language is underrepresented
    major_langs = ["en", "hi", "ta", "te", "bn", "mr", "gu", "pa"]
    underrepresented = [lang for lang in major_langs if language_counts.get(lang, 0) < 10]
    return LanguageStatsResponse(language_counts=language_counts, underrepresented=underrepresented)
    id: int
    consultation_id: Optional[str]
    law_section: Optional[str]
    original_text: str
    created_at: Optional[datetime]


@router.post("/submit", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def submit_comment(
    payload: CommentCreate,
    request: Request,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Public endpoint to submit a comment. If authenticated, link to user; otherwise store anonymously.
    """
    # Rate limit by client IP
    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "anonymous")
    _rate_limit_check(client_ip)

    # Spam/abuse heuristics and length requirement
    if not payload.text:
        raise HTTPException(status_code=400, detail="Comment text is required")
    spam_reason = _is_spammy(payload.text)
    if spam_reason:
        raise HTTPException(status_code=400, detail=spam_reason)

    comment = Comment(
        original_text=payload.text.strip(),
        law_section=payload.section,
        consultation_id=payload.draft_id,
        stakeholder_type=payload.stakeholder_type,
        stakeholder_category=payload.stakeholder_category,
        location=payload.location,
        submitted_at=payload.submitted_at or datetime.utcnow(),
        uploaded_by=current_user.id if current_user else None,
        word_count=len(payload.text.split()),
        character_count=len(payload.text),
    )

    db.add(comment)
    db.commit()
    db.refresh(comment)

    return CommentResponse(
        id=comment.id,
        consultation_id=comment.consultation_id,
        law_section=comment.law_section,
        original_text=comment.original_text,
        created_at=comment.created_at,
    )


@router.get("/list", response_model=List[Dict[str, Any]])
async def list_comments(
    consultation_id: Optional[str] = Query(None, description="Filter by consultation/draft ID"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Analysts/Admins can list stored comments. Guests require authentication.
    """
    if current_user.role not in [UserRole.ADMIN, UserRole.STAFF]:
        # guests can only view anonymized preview
        raise HTTPException(status_code=403, detail="Insufficient privileges to list comments")

    q = db.query(Comment)
    if consultation_id:
        q = q.filter(Comment.consultation_id == consultation_id)

    q = q.order_by(Comment.created_at.desc()).offset(offset).limit(limit)
    items = q.all()
    return [c.to_dict(include_text=False, include_analysis=False) for c in items]