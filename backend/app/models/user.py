"""
User model and schema for authentication.
"""

from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field
from enum import Enum

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

class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8, max_length=100)

class UserInDB(UserBase):
    """User model stored in DB."""
    hashed_password: str
    
    class Config:
        orm_mode = True

class User(UserBase):
    """User response model."""
    id: str
    
    class Config:
        orm_mode = True

class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None