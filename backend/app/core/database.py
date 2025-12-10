"""
Database connection and initialization for MongoDB.
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional

from backend.app.core.config import settings

# Add a placeholder Base for SQLAlchemy compatibility
class _SQLAlchemyBasePlaceholder:
    """Placeholder Base class for models that haven't been migrated to MongoDB yet."""
    pass

Base = _SQLAlchemyBasePlaceholder

logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB connection handler."""
    
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None
    _initialized = False

    @classmethod
    async def connect_db(cls) -> AsyncIOMotorDatabase:
        """Connect to MongoDB and return the database instance."""
        try:
            if cls._initialized and cls.client:
                return cls.db
                
            # Debug: Print the actual URI being used
            print(f"DEBUG: Connecting to MongoDB at {settings.MONGODB_URI}...")
            cls.client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=5000
            )
            
            # Test the connection
            await cls.client.admin.command('ping')
            
            # Set the database
            cls.db = cls.client[settings.MONGODB_DB]
            cls._initialized = True
            
            print(f"DEBUG: Connected to MongoDB successfully! Database: {settings.MONGODB_DB}")
            return cls.db
            
        except Exception as e:
            print(f"DEBUG: Failed to connect to MongoDB: {str(e)}")
            cls._initialized = False
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}") from e

    @classmethod
    async def close_db(cls):
        """Close the MongoDB connection."""
        if cls.client is not None:
            cls.client.close()
            cls.client = None
            cls.db = None
            cls._initialized = False
            logger.info("MongoDB connection closed.")

    @classmethod
    def get_db(cls) -> AsyncIOMotorDatabase:
        """Get the database instance."""
        if not cls._initialized or cls.db is None:
            raise RuntimeError("Database is not connected. Call connect_db() first.")
        return cls.db


async def get_db() -> AsyncIOMotorDatabase:
    """
    Dependency function to get MongoDB database instance.
    
    Returns:
        AsyncIOMotorDatabase: MongoDB database instance
    """
    try:
        db = await MongoDB.connect_db()
        return db
    except Exception as e:
        logger.error(f"Error getting database connection: {e}")
        raise


async def init_db():
    """Initialize the database with default data and indexes."""
    try:
        db = await MongoDB.connect_db()

        # Create indexes for better query performance
        await db.users.create_index("email", unique=True)
        await db.users.create_index("username", unique=True)

        logger.info("Database initialization completed (indexes ensured)")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def check_db_connection() -> bool:
    """Check if database connection is working."""
    try:
        db = await MongoDB.connect_db()
        # Try a simple ping
        await db.command('ping')
        return True
    except Exception:
        return False


async def get_db_info() -> dict:
    """Get database information and statistics."""
    try:
        db = await MongoDB.connect_db()
        stats = await db.command("dbStats")
        return {
            "connected": True,
            "database": settings.MONGODB_DB,
            "collections": stats.get("collections", 0),
            "objects": stats.get("objects", 0),
            "dataSize": stats.get("dataSize", 0),
            "indexSize": stats.get("indexSize", 0)
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }