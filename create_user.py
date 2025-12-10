import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from datetime import datetime

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)

async def create_sample_user():
    # MongoDB connection from env vars to avoid hard-coding secrets
    MONGODB_URI = os.environ.get("MONGODB_URI")
    MONGODB_DB = os.environ.get("MONGODB_DB", "sentiment_analysis")

    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI environment variable is required")
    
    client = AsyncIOMotorClient(MONGODB_URI)
    
    try:
        # Test the connection
        await client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        
        db = client[MONGODB_DB]
        
        # Check if users collection exists
        collections = await db.list_collection_names()
        if "users" not in collections:
            print("Creating users collection...")
            await db.create_collection("users")
        
        # Create indexes
        await db.users.create_index("email", unique=True)
        print("Created indexes on users collection")
        
        # Check if sample user already exists
        existing_user = await db.users.find_one({"email": "admin@example.com"})
        if existing_user:
            print("Sample user already exists")
            return
        
        # Create sample admin user
        sample_user = {
            "email": "admin@example.com",
            "username": "admin",
            "full_name": "Administrator",
            "hashed_password": get_password_hash("secret"),
            "role": "admin",
            "is_active": True,
            "is_verified": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await db.users.insert_one(sample_user)
        print(f"Created sample admin user with ID: {result.inserted_id}")
        
        # Create sample guest user
        sample_guest = {
            "email": "demo@econsult.gov",
            "username": "demo",
            "full_name": "Demo User",
            "hashed_password": get_password_hash("demo123"),
            "role": "staff",
            "is_active": True,
            "is_verified": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await db.users.insert_one(sample_guest)
        print(f"Created sample guest user with ID: {result.inserted_id}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(create_sample_user())