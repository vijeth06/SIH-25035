#!/usr/bin/env python3
"""
Database initialization script for E-Consultation Insight Engine.
Creates necessary directories and initializes the database.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent)
sys.path.append(project_root)

from app.core.config import settings
from app.core.database import Base

def init_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        settings.UPLOAD_DIR,
        settings.PROCESSED_DIR,
        settings.MODEL_CACHE_DIR,
        os.path.dirname(settings.LOG_FILE),
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Error creating directory {directory}: {e}")

def init_database():
    """Initialize the database."""
    try:
        # Create database if it doesn't exist
        if not database_exists(settings.DATABASE_URL):
            create_database(settings.DATABASE_URL)
            print(f"✓ Created database: {settings.DATABASE_URL}")
        else:
            print(f"✓ Database already exists: {settings.DATABASE_URL}")
        
        # Create tables
        engine = create_engine(settings.DATABASE_URL)
        Base.metadata.create_all(bind=engine)
        print("✓ Database tables created successfully")
        
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Initializing E-Consultation Insight Engine...")
    print(f"Project root: {project_root}")
    print("\nSetting up directories...")
    init_directories()
    
    print("\nInitializing database...")
    init_database()
    
    print("\n✅ Setup completed successfully!")
    print("\nYou can now start the application with:")
    print("1. Start the backend API:")
    print("   cd backend")
    print("   uvicorn app.main:app --reload")
    print("\n2. In a new terminal, start the Streamlit dashboard:")
    print("   cd dashboard")
    print("   streamlit run main.py")
    print("\n3. Open your browser to http://localhost:8501")
