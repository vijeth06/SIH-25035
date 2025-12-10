"""
MongoDB-specific Authentication API endpoints for user management and JWT tokens.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field

from backend.app.core.mongo_auth import authenticate_user, get_password_hash, get_user_by_email
from backend.app.core.security import create_access_token, create_refresh_token
from backend.app.core.database import MongoDB
from backend.app.models.mongo_models import UserInDB, UserRole, UserCreate
from backend.app.models.user import Token

router = APIRouter()

# Request/Response Models
class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    username: str
    full_name: Optional[str] = None
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserResponse

class TokenRefresh(BaseModel):
    """Token refresh request model."""
    refresh_token: str = Field(..., description="Refresh token")

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate):
    """
    Register a new user account.
    
    Creates a new user with the provided information. Email addresses must be unique.
    """
    db = await MongoDB.connect_db()
    
    # Check if user already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address already registered"
        )
    
    # Validate password strength
    if len(user_data.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    
    user_doc = {
        "email": user_data.email,
        "username": user_data.email,  # Use email as username
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "role": user_data.role.value,
        "is_active": True,
        "is_verified": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.users.insert_one(user_doc)
    user_doc["id"] = str(result.inserted_id)
    
    # Remove sensitive data
    user_doc.pop("hashed_password", None)
    
    return user_doc

@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT tokens.
    
    Uses OAuth2 password flow. Username should be the email address.
    Returns access token, refresh token, and user information.
    """
    user = await authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account is deactivated"
        )
    
    # Update last login
    db = await MongoDB.connect_db()
    await db.users.update_one(
        {"_id": user.id},
        {"$set": {"last_login": datetime.utcnow(), "updated_at": datetime.utcnow()}}
    )
    
    # Create tokens
    access_token_expires = timedelta(minutes=60 * 24 * 8)  # 8 days
    access_token = create_access_token(
        subject=user.email, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(subject=user.email)
    
    # Convert UserInDB to UserResponse
    user_response = UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        is_verified=user.is_verified,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=user.last_login
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=60 * 60 * 24 * 8,  # 8 days in seconds
        user=user_response
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info():
    """
    Get current user information.
    
    Returns the profile information of the currently authenticated user.
    """
    # This would normally use a JWT dependency to get the current user
    # For now, we'll return a placeholder
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Not implemented in this version"
    )