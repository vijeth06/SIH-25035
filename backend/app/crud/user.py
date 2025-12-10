"""
CRUD operations for user management with MongoDB.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from backend.app.core.database import MongoDB
from backend.app.models.mongo_models import UserInDB, UserCreate, UserUpdate, UserRole
from backend.app.core.security import get_password_hash

class CRUDUser:
    """CRUD operations for users."""

    @staticmethod
    async def get_by_id(user_id: str) -> Optional[UserInDB]:
        """Get a user by ID."""
        db = MongoDB.get_db()
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        return UserInDB(**user) if user else None

    @staticmethod
    async def get_by_email(email: str) -> Optional[UserInDB]:
        """Get a user by email."""
        db = MongoDB.get_db()
        user = await db.users.find_one({"email": email.lower()})
        return UserInDB(**user) if user else None

    @staticmethod
    async def get_by_username(username: str) -> Optional[UserInDB]:
        """Get a user by username."""
        db = MongoDB.get_db()
        user = await db.users.find_one({"username": username.lower()})
        return UserInDB(**user) if user else None

    @staticmethod
    async def create(user: UserCreate) -> UserInDB:
        """Create a new user."""
        db = MongoDB.get_db()
        hashed_password = get_password_hash(user.password)
        db_user = {
            **user.dict(exclude={"password"}),
            "hashed_password": hashed_password,
            "email": user.email.lower(),
            "username": user.username.lower(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True,
            "is_verified": False,
        }
        
        try:
            result = await db.users.insert_one(db_user)
            db_user["_id"] = result.inserted_id
            return UserInDB(**db_user)
        except DuplicateKeyError:
            # Check which field caused the duplicate key error
            existing = await db.users.find_one({
                "$or": [
                    {"email": user.email.lower()},
                    {"username": user.username.lower()}
                ]
            })
            if existing:
                if existing["email"] == user.email.lower():
                    raise ValueError("Email already registered")
                else:
                    raise ValueError("Username already taken")

    @staticmethod
    async def update(user_id: str, user_update: UserUpdate) -> Optional[UserInDB]:
        """Update a user."""
        db = MongoDB.get_db()
        update_data = user_update.dict(exclude_unset=True)
        
        if "password" in update_data:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
        
        if not update_data:
            return await CRUDUser.get_by_id(user_id)
            
        update_data["updated_at"] = datetime.utcnow()
        
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            return None
            
        return await CRUDUser.get_by_id(user_id)

    @staticmethod
    async def delete(user_id: str) -> bool:
        """Delete a user."""
        db = MongoDB.get_db()
        result = await db.users.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0

    @staticmethod
    async def authenticate(email: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user."""
        user = await CRUDUser.get_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    @staticmethod
    async def get_multi(
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[UserInDB]:
        """Get multiple users with optional filtering."""
        db = MongoDB.get_db()
        query = filters or {}
        
        cursor = db.users.find(query).skip(skip).limit(limit)
        return [UserInDB(**user) async for user in cursor]

    @staticmethod
    async def count(filters: Optional[Dict[str, Any]] = None) -> int:
        """Count users with optional filtering."""
        db = MongoDB.get_db()
        query = filters or {}
        return await db.users.count_documents(query)

# Create an instance of CRUDUser for easy import
user = CRUDUser()
