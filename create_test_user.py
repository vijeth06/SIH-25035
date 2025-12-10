"""
Script to create a test user for the sentiment analysis application.
"""

import os
import sys
from pathlib import Path

# Add the backend directory to the path
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

try:
    from backend.app.core.mongo_auth import get_password_hash
    from backend.app.core.database import MongoDB
    from backend.app.models.mongo_models import UserRole
    from datetime import datetime
    import asyncio
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you've run setup.bat first")
    sys.exit(1)

async def create_test_user():
    """Create a test user in the database."""
    try:
        # Connect to database
        db = await MongoDB.connect_db()
        
        # Check if user already exists
        existing_user = await db.users.find_one({"email": "test@example.com"})
        if existing_user:
            print("Test user already exists")
            return
        
        # Create test user
        hashed_password = get_password_hash("test123")
        
        user_doc = {
            "email": "test@example.com",
            "username": "test@example.com",
            "full_name": "Test User",
            "hashed_password": hashed_password,
            "role": UserRole.GUEST.value,
            "is_active": True,
            "is_verified": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await db.users.insert_one(user_doc)
        print(f"✅ Test user created successfully with ID: {result.inserted_id}")
        print("Login credentials:")
        print("  Email: test@example.com")
        print("  Password: test123")
        
    except Exception as e:
        print(f"❌ Error creating test user: {e}")

if __name__ == "__main__":
    print("Creating test user...")
    asyncio.run(create_test_user())