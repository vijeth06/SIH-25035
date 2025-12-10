"""
Initialize the database with demo users for testing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from datetime import datetime
from backend.app.core.database import get_db, create_tables
from backend.app.models.user import User, UserRole
from backend.app.core.security import get_password_hash


async def create_demo_users():
    """Create demo users for testing the application."""
    
    # Create tables first
    await create_tables()
    
    # Get database session
    db = next(get_db())
    
    # Demo users to create
    demo_users = [
        {
            "username": "demo_admin",
            "full_name": "Demo Admin",
            "email": "demo@econsult.gov",
            "password": "demo123",
            "role": UserRole.ADMIN
        },
        {
            "username": "demo_staff",
            "full_name": "Demo Staff",
            "email": "staff@econsult.gov", 
            "password": "staff123",
            "role": UserRole.STAFF
        },
        {
            "username": "demo_guest",
            "full_name": "Demo Guest",
            "email": "guest@econsult.gov",
            "password": "guest123", 
            "role": UserRole.GUEST
        }
    ]
    
    for user_data in demo_users:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data["email"]).first()
        
        if not existing_user:
            # Create new user
            hashed_password = get_password_hash(user_data["password"])
            
            new_user = User(
                username=user_data["username"],
                full_name=user_data["full_name"],
                email=user_data["email"],
                hashed_password=hashed_password,
                role=user_data["role"],
                is_active=True,
                created_at=datetime.utcnow()
            )
            
            db.add(new_user)
            print(f"✅ Created user: {user_data['email']} (Role: {user_data['role'].value})")
        else:
            print(f"👤 User already exists: {user_data['email']}")
    
    # Commit changes
    db.commit()
    db.close()
    
    print("\n🎉 Demo users initialization complete!")
    print("\n📋 Demo Credentials:")
    print("   👑 Admin: demo@econsult.gov / demo123")
    print("   👨‍💼 Staff: staff@econsult.gov / staff123") 
    print("   👤 Guest: guest@econsult.gov / guest123")


if __name__ == "__main__":
    print("🚀 Initializing demo users...")
    asyncio.run(create_demo_users())