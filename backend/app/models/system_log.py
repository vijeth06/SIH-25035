"""
System log model for tracking user activities and system events.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Enum as SQLAlchemyEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime

from backend.app.core.database import Base


class LogLevel(str, Enum):
    """Log levels for system events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogCategory(str, Enum):
    """Categories of system activities."""
    AUTH = "authentication"
    DATA_INGESTION = "data_ingestion"
    ANALYSIS = "analysis"
    USER_ACTION = "user_action"
    SYSTEM = "system"
    API = "api"
    SECURITY = "security"
    ERROR = "error"


class SystemLog(Base):
    """
    System log model for tracking activities and events.
    
    Provides comprehensive logging of user actions, system events,
    errors, and security-related activities for audit and monitoring.
    """
    
    __tablename__ = "system_logs"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    
    # Log classification
    level = Column(SQLAlchemyEnum(LogLevel), nullable=False, index=True)
    category = Column(SQLAlchemyEnum(LogCategory), nullable=False, index=True)
    
    # Log content
    message = Column(String(500), nullable=False)
    details = Column(Text, nullable=True)  # Detailed information
    log_metadata = Column(JSON, nullable=True)  # Additional structured data
    
    # Context information
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)  # Support IPv6
    user_agent = Column(String(500), nullable=True)
    
    # API and request context
    endpoint = Column(String(200), nullable=True)
    method = Column(String(10), nullable=True)  # GET, POST, etc.
    status_code = Column(Integer, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    
    # Resource and operation context
    resource_type = Column(String(50), nullable=True)  # comment, user, analysis, etc.
    resource_id = Column(Integer, nullable=True)
    operation = Column(String(50), nullable=True)  # create, read, update, delete, etc.
    
    # Error and exception information
    error_code = Column(String(50), nullable=True)
    exception_type = Column(String(100), nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    # Relationships
    user = relationship("User", backref="system_logs")
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level='{self.level}', category='{self.category}', message='{self.message[:50]}...')>"
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert system log to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive information like stack traces
            
        Returns:
            dict: System log data
        """
        result = {
            "id": self.id,
            "level": self.level.value,
            "category": self.category.value,
            "message": self.message,
            "details": self.details,
            "log_metadata": self.log_metadata,
            "user_id": self.user_id,
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "operation": self.operation,
            "error_code": self.error_code,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        
        if include_sensitive:
            result.update({
                "session_id": self.session_id,
                "ip_address": self.ip_address,
                "user_agent": self.user_agent,
                "exception_type": self.exception_type,
                "stack_trace": self.stack_trace,
            })
        else:
            # Anonymize sensitive data
            if self.ip_address:
                # Mask last octet of IP
                ip_parts = self.ip_address.split('.')
                if len(ip_parts) == 4:
                    result["ip_address"] = '.'.join(ip_parts[:3]) + '.xxx'
                else:
                    result["ip_address"] = "xxx.xxx.xxx.xxx"
            
            if self.user_agent:
                result["user_agent"] = "*** (masked)"
        
        return result
    
    @classmethod
    def log_auth_success(cls, user_id: int, ip_address: str = None, user_agent: str = None) -> 'SystemLog':
        """Create a log entry for successful authentication."""
        return cls(
            level=LogLevel.INFO,
            category=LogCategory.AUTH,
            message=f"User {user_id} authenticated successfully",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            operation="login"
        )
    
    @classmethod
    def log_auth_failure(cls, email: str, ip_address: str = None, reason: str = "invalid_credentials") -> 'SystemLog':
        """Create a log entry for failed authentication."""
        return cls(
            level=LogLevel.WARNING,
            category=LogCategory.SECURITY,
            message=f"Authentication failed for {email}",
            details=f"Reason: {reason}",
            ip_address=ip_address,
            operation="login_failed",
            metadata={"email": email, "reason": reason}
        )
    
    @classmethod
    def log_data_ingestion(cls, user_id: int, filename: str, records_processed: int, 
                          success: bool = True, error: str = None) -> 'SystemLog':
        """Create a log entry for data ingestion."""
        level = LogLevel.INFO if success else LogLevel.ERROR
        message = f"Data ingestion {'completed' if success else 'failed'}: {filename}"
        
        return cls(
            level=level,
            category=LogCategory.DATA_INGESTION,
            message=message,
            details=error if error else f"Processed {records_processed} records",
            user_id=user_id,
            operation="ingest",
            metadata={
                "filename": filename,
                "records_processed": records_processed,
                "success": success
            }
        )
    
    @classmethod
    def log_analysis_batch(cls, user_id: int, analysis_type: str, comment_count: int,
                          processing_time_ms: int, success: bool = True) -> 'SystemLog':
        """Create a log entry for batch analysis processing."""
        return cls(
            level=LogLevel.INFO,
            category=LogCategory.ANALYSIS,
            message=f"Batch {analysis_type} analysis completed for {comment_count} comments",
            user_id=user_id,
            operation="batch_analysis",
            response_time_ms=processing_time_ms,
            log_metadata={
                "analysis_type": analysis_type,
                "comment_count": comment_count,
                "processing_time_ms": processing_time_ms,
                "success": success
            }
        )
    
    @classmethod
    def log_api_request(cls, endpoint: str, method: str, status_code: int,
                       response_time_ms: int, user_id: int = None, ip_address: str = None) -> 'SystemLog':
        """Create a log entry for API requests."""
        level = LogLevel.INFO if status_code < 400 else LogLevel.WARNING if status_code < 500 else LogLevel.ERROR
        
        return cls(
            level=level,
            category=LogCategory.API,
            message=f"{method} {endpoint} -> {status_code}",
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            ip_address=ip_address,
            operation="api_request"
        )
    
    @classmethod
    def log_error(cls, message: str, details: str = None, user_id: int = None,
                 exception_type: str = None, stack_trace: str = None) -> 'SystemLog':
        """Create a log entry for system errors."""
        return cls(
            level=LogLevel.ERROR,
            category=LogCategory.ERROR,
            message=message,
            details=details,
            user_id=user_id,
            exception_type=exception_type,
            stack_trace=stack_trace,
            operation="error"
        )
    
    @classmethod
    def log_security_event(cls, event_type: str, message: str, user_id: int = None,
                          ip_address: str = None, metadata: dict = None) -> 'SystemLog':
        """Create a log entry for security events."""
        return cls(
            level=LogLevel.WARNING,
            category=LogCategory.SECURITY,
            message=message,
            user_id=user_id,
            ip_address=ip_address,
            operation=event_type,
            log_metadata=metadata
        )
    
    @classmethod
    def log_user_action(cls, user_id: int, action: str, resource_type: str = None,
                       resource_id: int = None, details: str = None) -> 'SystemLog':
        """Create a log entry for user actions."""
        return cls(
            level=LogLevel.INFO,
            category=LogCategory.USER_ACTION,
            message=f"User {user_id} performed {action}",
            details=details,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            operation=action
        )
    
    def is_error(self) -> bool:
        """Check if this is an error log entry."""
        return self.level in [LogLevel.ERROR, LogLevel.CRITICAL]
    
    def is_security_related(self) -> bool:
        """Check if this is a security-related log entry."""
        return self.category == LogCategory.SECURITY
    
    def involves_user(self, user_id: int) -> bool:
        """Check if this log entry involves a specific user."""
        return self.user_id == user_id