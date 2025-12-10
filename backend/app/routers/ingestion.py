"""
Data ingestion API endpoints for uploading and processing comment files.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from typing import Optional, Dict, Any

from backend.app.core.database import get_db
from backend.app.core.mongo_auth import get_current_user
from backend.app.services.ingestion_service import IngestionService
from backend.app.models.mongo_models import UserInDB as User

router = APIRouter()


@router.post("/upload", response_model=Dict[str, Any])
async def upload_file(
    file: UploadFile = File(..., description="File to upload (CSV, Excel, TXT, JSON)"),
    consultation_id: Optional[str] = Form(None, description="Consultation process ID"),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Upload and process a file containing stakeholder comments.

    Supports multiple file formats:
    - **CSV**: Comma-separated values with headers
    - **Excel**: .xlsx and .xls files (first sheet only)
    - **Text**: Plain text files (each line as a comment)
    - **JSON**: JSON files with comment objects

    The system will:
    - Validate the file format and content
    - Extract comments and metadata
    - Detect and mark duplicate comments
    - Store comments in the database
    - Return processing statistics
    """
    ingestion_service = IngestionService(db)

    try:
        result = await ingestion_service.process_upload(
            file=file,
            user_id=current_user.id,
            consultation_id=consultation_id
        )

        return {
            "success": True,
            "message": f"File processed successfully. {result['comments_saved']} comments saved.",
            "filename": file.filename,
            "statistics": result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_ingestion_stats(
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get data ingestion statistics for the current user or system-wide (for admins).
    """
    ingestion_service = IngestionService(db)

    # Admins can see system-wide stats, others see only their own
    user_id = None if hasattr(current_user, 'can_access_admin') and current_user.can_access_admin() else current_user.id

    stats = await ingestion_service.get_ingestion_stats(user_id=user_id)

    return {
        "success": True,
        "statistics": stats
    }


@router.post("/bulk-directory", response_model=Dict[str, Any])
async def bulk_process_directory(
    directory_path: str = Form(..., description="Path to directory containing files"),
    consultation_id: Optional[str] = Form(None, description="Consultation process ID"),
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Bulk process all supported files in a directory.

    **Note**: This endpoint is for server-side directory processing.
    The directory must be accessible to the server.
    """
    ingestion_service = IngestionService(db)

    try:
        result = await ingestion_service.bulk_process_directory(
            directory_path=directory_path,
            user_id=current_user.id,
            consultation_id=consultation_id
        )

        return {
            "success": True,
            "message": f"Bulk processing completed. {result['processed_files']}/{result['total_files']} files processed.",
            "statistics": result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in bulk processing: {str(e)}"
        )


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get information about supported file formats and requirements.
    """
    return {
        "supported_extensions": [".csv", ".xlsx", ".xls", ".txt", ".json"],
        "max_file_size_mb": 50,
        "formats": {
            "csv": {
                "description": "Comma-separated values file with headers",
                "expected_columns": {
                    "required": ["comment", "text", "feedback", "response", "content"],
                    "optional": ["id", "law_section", "stakeholder_type", "location", "submitted_at"]
                },
                "encoding": "UTF-8 recommended, Latin-1 and CP1252 also supported"
            },
            "excel": {
                "description": "Excel spreadsheet (.xlsx or .xls)",
                "notes": "Only the first sheet will be processed",
                "expected_columns": "Same as CSV format"
            },
            "text": {
                "description": "Plain text file with one comment per line",
                "encoding": "UTF-8"
            },
            "json": {
                "description": "JSON file with array of comment objects",
                "structure": "Array of objects or object with 'comments'/'data' array"
            }
        },
        "column_mapping": {
            "comment_text": ["comment", "text", "feedback", "response", "content", "message"],
            "comment_id": ["id", "comment_id", "feedback_id", "response_id"],
            "law_section": ["law_section", "section", "article", "clause"],
            "stakeholder_info": ["stakeholder_type", "user_type", "respondent_type"],
            "location": ["location", "city", "state", "region"],
            "timestamp": ["submitted_at", "date", "timestamp", "created_at"]
        }
    }