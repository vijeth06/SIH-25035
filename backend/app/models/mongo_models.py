"""
MongoDB models for the application using Motor.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr

class UserRole(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    STAFF = "staff"
    GUEST = "guest"

class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.GUEST
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8, max_length=100)

class UserInDB(UserBase):
    """User model stored in DB."""
    id: str = Field(..., alias="_id")
    hashed_password: str
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserUpdate(BaseModel):
    """User update model."""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    password: Optional[str] = None

class AnalysisResult(BaseModel):
    """Analysis result model."""
    id: str = Field(..., alias="_id")
    user_id: str
    text: str
    sentiment: Dict[str, float]
    entities: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CommentBase(BaseModel):
    """Base comment model."""
    original_text: str = Field(..., min_length=1, max_length=10000)
    processed_text: Optional[str] = None
    source_file: str
    source_row: Optional[int] = None
    comment_id_external: Optional[str] = None
    law_section: Optional[str] = None
    consultation_id: Optional[str] = None
    stakeholder_type: Optional[str] = None
    stakeholder_category: Optional[str] = None
    location: Optional[str] = None
    submitted_at: Optional[datetime] = None
    uploaded_by: str  # User ID
    word_count: int = Field(default=0)
    character_count: int = Field(default=0)
    is_duplicate: bool = Field(default=False)
    duplicate_of: Optional[str] = None  # Comment ID
    sentiment_analysis: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class CommentInDB(CommentBase):
    """Comment model stored in DB."""
    id: str = Field(..., alias="_id")

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CommentCreate(CommentBase):
    """Comment creation model."""
    pass

class SystemLogBase(BaseModel):
    """Base system log model."""
    user_id: Optional[str] = None
    action: str
    resource: str
    resource_id: Optional[str] = None
    details: Dict[str, Any] = {}
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SystemLogInDB(SystemLogBase):
    """System log model stored in DB."""
    id: str = Field(..., alias="_id")

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
