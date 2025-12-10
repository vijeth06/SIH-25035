"""
MongoDB-specific authentication utilities.
"""

from datetime import datetime
from typing import Optional
from passlib.context import CryptContext
from jose import jwt, JWTError
from backend.app.core.database import MongoDB
from backend.app.models.mongo_models import UserInDB
from backend.app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)

async def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user against MongoDB."""
    try:
        # Ensure database connection
        db = await MongoDB.connect_db()
        user_data = await db.users.find_one({"email": email})
        if user_data and verify_password(password, user_data.get("hashed_password", "")):
            # Convert ObjectId to string and set it as _id for Pydantic model
            user_data["_id"] = str(user_data["_id"])
            return UserInDB(**user_data)
        return None
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

async def get_user_by_email(email: str) -> Optional[UserInDB]:
    """Get a user by email from MongoDB."""
    try:
        # Ensure database connection
        db = await MongoDB.connect_db()
        user_data = await db.users.find_one({"email": email})
        if user_data:
            # Convert ObjectId to string and set it as _id for Pydantic model
            user_data["_id"] = str(user_data["_id"])
            return UserInDB(**user_data)
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

from fastapi import Header, HTTPException, status

async def get_current_user(authorization: str = Header(...)) -> UserInDB:
    """Get current user from JWT token in Authorization header for MongoDB implementation."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = await get_user_by_email(email)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
from fastapi import Header, Depends

async def get_optional_current_user(authorization: Optional[str] = Header(None)) -> Optional[UserInDB]:
    """
    Returns the current user if a valid Bearer token is provided.
    Returns None if no token is provided or token is invalid.
    """
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return await get_current_user(token)
    return None
