"""
Health check router for monitoring application status.
"""

from fastapi import APIRouter, Depends
from datetime import datetime
from sqlalchemy.orm import Session

from backend.app.core.database import get_db, check_db_connection, get_db_info
from backend.app.core.config import settings


router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        dict: Application health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "app_name": settings.APP_NAME,
    }


@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Detailed health check with database and system status.
    
    Args:
        db: Database session
        
    Returns:
        dict: Detailed system health information
    """
    # Check database connection
    db_healthy = check_db_connection()
    db_info = get_db_info() if db_healthy else {"status": "error"}
    
    # System status
    system_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "debug_mode": settings.DEBUG,
        "database": db_info,
        "services": {
            "api": "healthy",
            "database": "healthy" if db_healthy else "unhealthy",
            "nlp_models": "loading",  # Will be updated when NLP services are added
        },
        "configuration": {
            "supported_languages": settings.SUPPORTED_LANGUAGES,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE // (1024 * 1024),
            "allowed_extensions": settings.ALLOWED_EXTENSIONS,
        }
    }
    
    # Overall health status
    overall_status = "healthy" if db_healthy else "degraded"
    
    return {
        "status": overall_status,
        **system_status
    }


@router.get("/health/database")
async def database_health():
    """
    Database-specific health check.
    
    Returns:
        dict: Database connection and statistics
    """
    db_healthy = check_db_connection()
    db_info = get_db_info() if db_healthy else None
    
    return {
        "database_healthy": db_healthy,
        "database_info": db_info,
        "timestamp": datetime.utcnow().isoformat(),
    }