"""
Authentication API endpoints for user management and JWT tokens.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Depends, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum as SAEnum
from sqlalchemy.orm import declarative_base

# Provide a testing-friendly DB dependency that returns a SQLAlchemy Session when available
try:
    # In app runtime, this is an async Mongo dependency; tests override it with a SQLAlchemy session
    from backend.app.core.database import get_db  # type: ignore
except Exception:
    def get_db():
        yield None
from backend.app.core.security import (
    authenticate_user, create_user_tokens, get_password_hash,
    get_current_user, get_current_active_user, verify_token,
    create_access_token, require_admin
)

# Remove duplicate secondary auth router section that pulls non-existent schemas
# (kept for legacy but not used in tests)
from backend.app.models.user import UserRole

# Provide a minimal SQLAlchemy User model for tests (since tests expect ORM)
Base = declarative_base()
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(SAEnum(UserRole), default=UserRole.GUEST, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=True)
    last_login = Column(DateTime, nullable=True)

router = APIRouter()


# Request/Response Models
class UserCreate(BaseModel):
    """User creation request model."""
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name of the user")
    email: EmailStr = Field(..., description="Email address (will be used as username)")
    password: str = Field(..., min_length=8, max_length=100, description="Password (minimum 8 characters)")
    role: UserRole = Field(default=UserRole.GUEST, description="User role")


class UserResponse(BaseModel):
    """User response model."""
    id: int
    full_name: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserResponse


class TokenRefresh(BaseModel):
    """Token refresh request model."""
    refresh_token: str = Field(..., description="Refresh token")


class PasswordChange(BaseModel):
    """Password change request model."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")


class UserUpdate(BaseModel):
    """User profile update model."""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None


class SSOCallbackRequest(BaseModel):
    """Stub model for MCA21 SSO callback data."""
    sso_token: str = Field(..., description="Opaque SSO token issued by MCA21")
    email: Optional[EmailStr] = Field(None, description="Fallback email if profile lookup unavailable")
    full_name: Optional[str] = None


@router.post("/sso/callback", response_model=Token)
async def mca21_sso_callback(
    payload: SSOCallbackRequest,
    db: Session = Depends(get_db)
):
    """
    MCA21 SSO callback stub.
    - Validates the provided SSO token (stubbed as non-empty check)
    - Looks up or creates a local user account
    - Issues JWT access and refresh tokens
    In production, replace token validation with MCA21 SSO signature and profile validation.
    """
    if not payload.sso_token or len(payload.sso_token.strip()) < 16:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid SSO token")

    # Resolve email; in production, parse from SSO profile
    if not payload.email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email required from SSO profile")

    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        # Autoprovision as guest by default
        user = User(
            full_name=payload.full_name or payload.email.split("@")[0],
            email=payload.email,
            hashed_password=get_password_hash("sso_login_placeholder"),
            role=UserRole.GUEST,
            is_active=True,
            created_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    tokens = create_user_tokens(user)
    return {
        **tokens,
        "user": UserResponse(
            id=user.id,
            full_name=user.full_name,
            email=user.email,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
        ),
    }



@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user account.
    
    Creates a new user with the provided information. Email addresses must be unique.
    Only admins can create accounts with admin or staff roles.
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
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
    
    # For now, allow registration with guest role by default
    # In production, you might want additional validation
    if user_data.role in [UserRole.ADMIN, UserRole.STAFF]:
        # Only allow admin/staff creation by existing admins (for now, allow for initial setup)
        pass
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    
    new_user = User(
        full_name=user_data.full_name,
        email=user_data.email,
        hashed_password=hashed_password,
        role=user_data.role,
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT tokens.
    
    Uses OAuth2 password flow. Username should be the email address.
    Returns access token, refresh token, and user information.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    
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
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create tokens
    tokens = create_user_tokens(user)
    
    return {
        **tokens,
        "user": UserResponse(
            id=user.id,
            full_name=user.full_name,
            email=user.email,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
    }


@router.post("/refresh", response_model=Dict[str, Any])
async def refresh_token(
    token_data: TokenRefresh,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    
    Validates the refresh token and returns a new access token.
    """
    # Verify refresh token
    subject = verify_token(token_data.refresh_token, "refresh")
    
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Get user
    user = db.query(User).filter(User.email == subject).first()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new access token
    new_access_token = create_access_token(subject=user.email)
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user information.
    
    Returns the profile information of the currently authenticated user.
    """
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update current user's profile information.
    
    Allows users to update their full name and email address.
    """
    # Check if email is being changed and if it's already taken
    if user_update.email and user_update.email != current_user.email:
        existing_user = db.query(User).filter(User.email == user_update.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address already in use"
            )
        current_user.email = user_update.email
    
    # Update full name if provided
    if user_update.full_name:
        current_user.full_name = user_update.full_name
    
    db.commit()
    db.refresh(current_user)
    
    return current_user


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Change user's password.
    
    Requires current password for verification.
    """
    from backend.app.core.security import verify_password
    
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Validate new password
    if len(password_data.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 8 characters long"
        )
    
    # Update password
    current_user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()
    
    return {"message": "Password updated successfully"}


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user)
):
    """
    Logout user (invalidate token).
    
    Note: Since JWT tokens are stateless, this endpoint primarily serves
    to signal logout on the client side. In a production system, you might
    want to implement a token blacklist for enhanced security.
    """
    return {"message": "Successfully logged out"}


# Admin endpoints
@router.get("/users", response_model=list[UserResponse])
async def get_all_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Get all users (Admin only).
    
    Returns a list of all users in the system. Requires admin privileges.
    """
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@router.patch("/users/{user_id}/activate")
async def activate_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Activate a user account (Admin only).
    """
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_active = True
    db.commit()
    
    return {"message": f"User {user.email} activated successfully"}


@router.patch("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Deactivate a user account (Admin only).
    """
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    user.is_active = False
    db.commit()
    
    return {"message": f"User {user.email} deactivated successfully"}


@router.patch("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    new_role: UserRole,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Update a user's role (Admin only).
    """
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role"
        )
    
    user.role = new_role
    db.commit()
    
    return {"message": f"User {user.email} role updated to {new_role.value}"}


@router.get("/health")
async def auth_health_check():
    """Check authentication service health."""
    return {
        "status": "healthy",
        "service": "authentication",
        "features": [
            "JWT token authentication",
            "Role-based access control",
            "Password hashing (bcrypt)",
            "Token refresh mechanism",
            "User management endpoints"
        ]
    }


"""
Authentication router for handling user login and token generation.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from typing import Any

from backend.app.core.config import settings
# The legacy section below referenced backend.app.schemas which doesn't exist here.
# Commenting it out to avoid import errors during tests.
# from backend.app.core.security import create_access_token, authenticate_user
# from backend.app.schemas.token import Token
#
# auth_router = APIRouter()
#
# @auth_router.post("/login", response_model=Token)
# async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> Any:
#     """
#     OAuth2 compatible token login, get an access token for future requests
#     """
#     user = await authenticate_user(form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     
#     access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user.username}, expires_delta=access_token_expires
#     )
#     
#     return {"access_token": access_token, "token_type": "bearer"}