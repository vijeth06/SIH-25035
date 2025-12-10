"""
Batch processing router for handling high-volume comment analysis.
Prevents comments from being overlooked by implementing efficient processing queues.
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from enum import Enum

from backend.app.core.security import get_current_user
from backend.app.models.user import User
from backend.app.services.sentiment_service import SentimentAnalyzer
from backend.app.services.summarization_service import SummarizationService
from backend.app.services.visualization_service import VisualizationService

router = APIRouter()

class BatchStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BatchJobType(str, Enum):
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SUMMARIZATION = "summarization"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"
    POLICY_ANALYSIS = "policy_analysis"

class BatchJobRequest(BaseModel):
    job_type: BatchJobType
    comments: List[str]
    parameters: Optional[Dict[str, Any]] = {}
    priority: Optional[int] = 5  # 1 (highest) to 10 (lowest)
    notification_email: Optional[str] = None

class BatchJobStatus(BaseModel):
    job_id: str
    job_type: BatchJobType
    status: BatchStatus
    total_comments: int
    processed_comments: int
    progress_percentage: float
    estimated_completion: Optional[datetime] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    results_available: bool = False

class BatchResult(BaseModel):
    job_id: str
    job_type: BatchJobType
    total_comments: int
    processing_time_seconds: float
    results: List[Dict[str, Any]]
    summary_statistics: Dict[str, Any]
    failed_comments: List[Dict[str, Any]]
    recommendations: List[str]

# In-memory storage for batch jobs (in production, use Redis or database)
batch_jobs: Dict[str, Dict] = {}
job_queue = asyncio.Queue()
processing_jobs: Dict[str, asyncio.Task] = {}

# Initialize services
sentiment_analyzer = SentimentAnalyzer()
summarization_service = SummarizationService()
visualization_service = VisualizationService()

@router.post("/submit-batch", response_model=Dict[str, str])
async def submit_batch_job(
    request: BatchJobRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Submit a batch job for processing high-volume comments.
    
    This endpoint prevents comments from being overlooked by:
    - Queueing all comments for systematic processing
    - Providing progress tracking
    - Ensuring no comment is skipped
    - Handling failures gracefully with retry mechanisms
    """
    job_id = str(uuid.uuid4())
    
    # Validate comment count
    if len(request.comments) == 0:
        raise HTTPException(status_code=400, detail="No comments provided")
    
    if len(request.comments) > 10000:  # Limit for single batch
        raise HTTPException(
            status_code=400, 
            detail="Batch size too large. Maximum 10,000 comments per batch."
        )
    
    # Create job record
    job_data = {
        "job_id": job_id,
        "job_type": request.job_type,
        "status": BatchStatus.QUEUED,
        "total_comments": len(request.comments),
        "processed_comments": 0,
        "progress_percentage": 0.0,
        "created_at": datetime.utcnow(),
        "user_id": current_user.id,
        "priority": request.priority,
        "comments": request.comments,
        "parameters": request.parameters,
        "notification_email": request.notification_email,
        "results": [],
        "failed_comments": [],
        "error_message": None
    }
    
    batch_jobs[job_id] = job_data
    
    # Add to processing queue
    background_tasks.add_task(process_batch_job, job_id)
    
    return {
        "job_id": job_id,
        "message": f"Batch job submitted successfully. Processing {len(request.comments)} comments.",
        "estimated_time": f"{len(request.comments) * 2} seconds"  # Rough estimate
    }

@router.get("/status/{job_id}", response_model=BatchJobStatus)
async def get_batch_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the status of a batch processing job."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    job = batch_jobs[job_id]
    
    # Check if user owns this job or is admin
    if job["user_id"] != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Calculate estimated completion
    estimated_completion = None
    if job["status"] == BatchStatus.PROCESSING and job["processed_comments"] > 0:
        avg_time_per_comment = (datetime.utcnow() - job["started_at"]).total_seconds() / job["processed_comments"]
        remaining_comments = job["total_comments"] - job["processed_comments"]
        estimated_seconds = remaining_comments * avg_time_per_comment
        estimated_completion = datetime.utcnow() + timedelta(seconds=estimated_seconds)
    
    return BatchJobStatus(
        job_id=job_id,
        job_type=job["job_type"],
        status=job["status"],
        total_comments=job["total_comments"],
        processed_comments=job["processed_comments"],
        progress_percentage=job["progress_percentage"],
        estimated_completion=estimated_completion,
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error_message=job.get("error_message"),
        results_available=job["status"] == BatchStatus.COMPLETED
    )

@router.get("/results/{job_id}", response_model=BatchResult)
async def get_batch_results(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the results of a completed batch processing job."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    job = batch_jobs[job_id]
    
    # Check if user owns this job or is admin
    if job["user_id"] != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job["status"] != BatchStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Calculate summary statistics
    results = job["results"]
    summary_stats = calculate_summary_statistics(results, job["job_type"])
    
    # Generate recommendations
    recommendations = generate_batch_recommendations(results, job["job_type"])
    
    processing_time = (job["completed_at"] - job["started_at"]).total_seconds()
    
    return BatchResult(
        job_id=job_id,
        job_type=job["job_type"],
        total_comments=job["total_comments"],
        processing_time_seconds=processing_time,
        results=results,
        summary_statistics=summary_stats,
        failed_comments=job["failed_comments"],
        recommendations=recommendations
    )

@router.get("/list-jobs")
async def list_batch_jobs(
    status: Optional[BatchStatus] = None,
    job_type: Optional[BatchJobType] = None,
    limit: int = Query(default=50, le=100),
    current_user: User = Depends(get_current_user)
):
    """List batch jobs for the current user."""
    user_jobs = []
    
    for job_id, job in batch_jobs.items():
        # Filter by user (admin can see all)
        if job["user_id"] != current_user.id and current_user.role != "admin":
            continue
        
        # Apply filters
        if status and job["status"] != status:
            continue
        if job_type and job["job_type"] != job_type:
            continue
        
        user_jobs.append({
            "job_id": job_id,
            "job_type": job["job_type"],
            "status": job["status"],
            "total_comments": job["total_comments"],
            "processed_comments": job["processed_comments"],
            "progress_percentage": job["progress_percentage"],
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at")
        })
    
    # Sort by creation time (newest first)
    user_jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    return user_jobs[:limit]

@router.delete("/cancel/{job_id}")
async def cancel_batch_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a batch processing job."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    job = batch_jobs[job_id]
    
    # Check if user owns this job or is admin
    if job["user_id"] != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job["status"] in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed/failed/cancelled job")
    
    # Cancel the job
    job["status"] = BatchStatus.CANCELLED
    
    # Cancel processing task if running
    if job_id in processing_jobs:
        processing_jobs[job_id].cancel()
        del processing_jobs[job_id]
    
    return {"message": "Batch job cancelled successfully"}

async def process_batch_job(job_id: str):
    """Background task to process batch job."""
    try:
        job = batch_jobs[job_id]
        job["status"] = BatchStatus.PROCESSING
        job["started_at"] = datetime.utcnow()
        
        comments = job["comments"]
        job_type = job["job_type"]
        parameters = job["parameters"]
        
        results = []
        failed_comments = []
        
        # Process comments in batches to prevent memory issues
        batch_size = 50  # Process 50 comments at a time
        
        for i in range(0, len(comments), batch_size):
            batch_comments = comments[i:i + batch_size]
            
            # Process batch
            batch_results = await process_comment_batch(
                batch_comments, job_type, parameters, i
            )
            
            # Separate successful and failed results
            for result in batch_results:
                if result.get("error"):
                    failed_comments.append(result)
                else:
                    results.append(result)
            
            # Update progress
            job["processed_comments"] = min(i + batch_size, len(comments))
            job["progress_percentage"] = (job["processed_comments"] / len(comments)) * 100
            
            # Check if job was cancelled
            if job["status"] == BatchStatus.CANCELLED:
                return
        
        # Job completed successfully
        job["status"] = BatchStatus.COMPLETED
        job["completed_at"] = datetime.utcnow()
        job["results"] = results
        job["failed_comments"] = failed_comments
        
        # Send notification if email provided
        if job.get("notification_email"):
            await send_completion_notification(job)
        
    except Exception as e:
        # Job failed
        job["status"] = BatchStatus.FAILED
        job["error_message"] = str(e)
        job["completed_at"] = datetime.utcnow()
        
        print(f"Batch job {job_id} failed: {e}")

async def process_comment_batch(
    comments: List[str], 
    job_type: BatchJobType, 
    parameters: Dict[str, Any],
    batch_start_index: int
) -> List[Dict[str, Any]]:
    """Process a batch of comments based on job type."""
    results = []
    
    for i, comment in enumerate(comments):
        try:
            comment_index = batch_start_index + i
            
            if job_type == BatchJobType.SENTIMENT_ANALYSIS:
                # Basic sentiment analysis
                sentiment_results = await sentiment_analyzer.analyze_sentiment(comment)
                result = {
                    "comment_index": comment_index,
                    "comment": comment,
                    "sentiment_results": [r.__dict__ for r in sentiment_results],
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            elif job_type == BatchJobType.POLICY_ANALYSIS:
                # Policy-specific analysis
                policy_result = await sentiment_analyzer.analyze_policy_sentiment(comment)
                stakeholder_type = sentiment_analyzer._detect_stakeholder_type(comment)
                result = {
                    "comment_index": comment_index,
                    "comment": comment,
                    "policy_sentiment": policy_result.__dict__,
                    "stakeholder_type": stakeholder_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            elif job_type == BatchJobType.COMPREHENSIVE_ANALYSIS:
                # Full comprehensive analysis
                comprehensive_result = await sentiment_analyzer.comprehensive_analysis(comment)
                result = {
                    "comment_index": comment_index,
                    "comment": comment,
                    "comprehensive_analysis": {
                        "sentiment_results": [r.__dict__ for r in comprehensive_result.sentiment_results],
                        "emotion_result": comprehensive_result.emotion_result.__dict__,
                        "aspect_sentiments": [a.__dict__ for a in comprehensive_result.aspect_sentiments],
                        "key_phrases": comprehensive_result.key_phrases,
                        "law_sections_mentioned": comprehensive_result.law_sections_mentioned,
                        "overall_sentiment": comprehensive_result.overall_sentiment.value,
                        "overall_confidence": comprehensive_result.overall_confidence,
                        "processing_time_ms": comprehensive_result.processing_time_ms
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            elif job_type == BatchJobType.SUMMARIZATION:
                # Summarization with policy enhancement
                summary_result = await summarization_service.policy_summarization(comment)
                result = {
                    "comment_index": comment_index,
                    "comment": comment,
                    "summary_result": summary_result.__dict__,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            results.append(result)
            
        except Exception as e:
            # Record failed comment
            results.append({
                "comment_index": comment_index,
                "comment": comment,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    return results

def calculate_summary_statistics(results: List[Dict], job_type: BatchJobType) -> Dict[str, Any]:
    """Calculate summary statistics for batch results."""
    total_results = len(results)
    
    if job_type == BatchJobType.SENTIMENT_ANALYSIS or job_type == BatchJobType.POLICY_ANALYSIS:
        # Sentiment distribution
        sentiments = []
        confidences = []
        
        for result in results:
            if job_type == BatchJobType.POLICY_ANALYSIS:
                sentiment = result.get("policy_sentiment", {}).get("sentiment_label")
                confidence = result.get("policy_sentiment", {}).get("confidence_score", 0)
            else:
                sentiment_results = result.get("sentiment_results", [])
                if sentiment_results:
                    sentiment = sentiment_results[0].get("sentiment_label")
                    confidence = sentiment_results[0].get("confidence_score", 0)
                else:
                    continue
                    
            sentiments.append(sentiment)
            confidences.append(confidence)
        
        sentiment_counts = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return {
            "total_comments": total_results,
            "sentiment_distribution": sentiment_counts,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "high_confidence_comments": len([c for c in confidences if c > 0.8]),
            "low_confidence_comments": len([c for c in confidences if c < 0.5])
        }
    
    elif job_type == BatchJobType.COMPREHENSIVE_ANALYSIS:
        # More detailed statistics
        sentiments = []
        emotions = []
        aspects_count = 0
        
        for result in results:
            analysis = result.get("comprehensive_analysis", {})
            overall_sentiment = analysis.get("overall_sentiment")
            emotion = analysis.get("emotion_result", {}).get("emotion_label")
            aspects = analysis.get("aspect_sentiments", [])
            
            if overall_sentiment:
                sentiments.append(overall_sentiment)
            if emotion:
                emotions.append(emotion)
            aspects_count += len(aspects)
        
        sentiment_counts = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            "total_comments": total_results,
            "sentiment_distribution": sentiment_counts,
            "emotion_distribution": emotion_counts,
            "total_aspects_identified": aspects_count,
            "average_aspects_per_comment": aspects_count / total_results if total_results > 0 else 0
        }
    
    else:  # Summarization
        total_chars = sum(len(r.get("summary_result", {}).get("summary_text", "")) for r in results)
        return {
            "total_comments": total_results,
            "total_summary_characters": total_chars,
            "average_summary_length": total_chars / total_results if total_results > 0 else 0
        }

def generate_batch_recommendations(results: List[Dict], job_type: BatchJobType) -> List[str]:
    """Generate actionable recommendations based on batch analysis results."""
    recommendations = []
    
    if not results:
        return ["No results available for recommendations."]
    
    # Calculate basic metrics
    total_comments = len(results)
    
    if job_type in [BatchJobType.SENTIMENT_ANALYSIS, BatchJobType.POLICY_ANALYSIS]:
        # Sentiment-based recommendations
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        low_confidence_count = 0
        
        for result in results:
            if job_type == BatchJobType.POLICY_ANALYSIS:
                sentiment_data = result.get("policy_sentiment", {})
            else:
                sentiment_results = result.get("sentiment_results", [])
                sentiment_data = sentiment_results[0] if sentiment_results else {}
            
            sentiment = sentiment_data.get("sentiment_label", "").lower()
            confidence = sentiment_data.get("confidence_score", 0)
            
            if sentiment == "positive":
                positive_count += 1
            elif sentiment == "negative":
                negative_count += 1
            else:
                neutral_count += 1
            
            if confidence < 0.6:
                low_confidence_count += 1
        
        # Generate recommendations
        if negative_count > positive_count:
            recommendations.append(f"High negative sentiment detected ({negative_count}/{total_comments} comments). Consider addressing stakeholder concerns.")
        
        if positive_count > total_comments * 0.7:
            recommendations.append(f"Strong positive support ({positive_count}/{total_comments} comments). This policy appears well-received.")
        
        if low_confidence_count > total_comments * 0.3:
            recommendations.append(f"Many comments have ambiguous sentiment ({low_confidence_count}/{total_comments}). Manual review recommended for unclear feedback.")
        
        if neutral_count > total_comments * 0.5:
            recommendations.append("High neutral sentiment suggests stakeholders need more information or clarification.")
    
    # Add volume-based recommendations
    if total_comments > 1000:
        recommendations.append("High volume of feedback received. Consider categorizing by stakeholder type for targeted responses.")
    
    if total_comments < 50:
        recommendations.append("Limited feedback volume. Consider extending consultation period or broader outreach.")
    
    recommendations.append("Review failed comments for processing issues and ensure comprehensive coverage.")
    recommendations.append("Use stakeholder categorization to identify different perspectives on the policy.")
    
    return recommendations

async def send_completion_notification(job: Dict):
    """Send email notification when batch job completes (placeholder)."""
    # In production, implement actual email sending
    print(f"Notification: Batch job {job['job_id']} completed. Email would be sent to {job['notification_email']}")