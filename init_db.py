#!/usr/bin/env python3
"""
Database Initialization Script

This script initializes the MongoDB database with required collections and sample data.
"""

import os
import sys
import logging
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env file."""
    env_path = Path('.') / '.env'
    if not env_path.exists():
        logger.error("Error: .env file not found in the current directory.")
        logger.info("Please create a .env file with your MongoDB connection string.")
        return False
    
    load_dotenv(dotenv_path=env_path)
    
    if not os.getenv('MONGODB_URI'):
        logger.error("Error: MONGODB_URI not found in .env file.")
        logger.info("Please add MONGODB_URI to your .env file.")
        return False
    
    return True

def initialize_database():
    """Initialize the MongoDB database with required collections and sample data."""
    try:
        # Connect to MongoDB
        client = MongoClient(os.getenv('MONGODB_URI'))
        db_name = os.getenv('MONGODB_DB', 'sentiment_analysis')
        db = client[db_name]
        
        # Test the connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")
        
        # Define collections and their schemas
        collections = {
            'users': {
                'validator': {
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['email', 'hashed_password', 'is_active', 'created_at'],
                        'properties': {
                            'email': {'bsonType': 'string'},
                            'hashed_password': {'bsonType': 'string'},
                            'is_active': {'bsonType': 'bool'},
                            'created_at': {'bsonType': 'date'}
                        }
                    }
                },
                'indexes': [
                    [('email', 1), {'unique': True}]
                ]
            },
            'comments': {
                'validator': {
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['text', 'sentiment', 'created_at'],
                        'properties': {
                            'text': {'bsonType': 'string'},
                            'sentiment': {
                                'bsonType': 'object',
                                'required': ['label', 'score'],
                                'properties': {
                                    'label': {'bsonType': 'string'},
                                    'score': {'bsonType': 'double'}
                                }
                            },
                            'created_at': {'bsonType': 'date'}
                        }
                    }
                },
                'indexes': [
                    [('created_at', -1)],
                    [('sentiment.label', 1)]
                ]
            },
            'analyses': {
                'validator': {
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['user_id', 'text', 'sentiment', 'created_at'],
                        'properties': {
                            'user_id': {'bsonType': 'objectId'},
                            'text': {'bsonType': 'string'},
                            'sentiment': {
                                'bsonType': 'object',
                                'required': ['label', 'score'],
                                'properties': {
                                    'label': {'bsonType': 'string'},
                                    'score': {'bsonType': 'double'}
                                }
                            },
                            'created_at': {'bsonType': 'date'}
                        }
                    }
                },
                'indexes': [
                    [('user_id', 1)],
                    [('created_at', -1)]
                ]
            }
        }
        
        # Create collections and indexes
        for coll_name, config in collections.items():
            if coll_name not in db.list_collection_names():
                db.create_collection(coll_name)
                logger.info(f"Created collection: {coll_name}")
            
            # Update validation
            db.command({
                'collMod': coll_name,
                'validator': config['validator'],
                'validationLevel': 'strict',
                'validationAction': 'error'
            })
            
            # Create indexes
            for index_spec in config.get('indexes', []):
                if isinstance(index_spec, list):
                    # Handle list format: [('field', 1), {'unique': True}]
                    fields = index_spec[0] if isinstance(index_spec[0], list) else [index_spec[0]]
                    options = index_spec[1] if len(index_spec) > 1 else {}
                    db[coll_name].create_index(fields, **options)
                else:
                    # Handle direct specification: ('field', 1)
                    db[coll_name].create_index(index_spec)
            logger.info(f"Configured collection: {coll_name}")
        
        # Add sample data
        if db.users.count_documents({}) == 0:
            from datetime import datetime
            from passlib.context import CryptContext
            
            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            sample_user = {
                'email': 'admin@example.com',
                'hashed_password': pwd_context.hash('secret'),
                'is_active': True,
                'is_superuser': True,
                'created_at': datetime.utcnow()
            }
            db.users.insert_one(sample_user)
            logger.info("Added sample admin user (email: admin@example.com, password: secret)")
        
        logger.info("Database initialization completed successfully!")
        return True
        
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return False
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return False

def main():
    """Main function to run the database initialization."""
    if not load_environment():
        return 1
    
    logger.info("Starting database initialization...")
    if not initialize_database():
        logger.error("Database initialization failed.")
        return 1
    
    logger.info("\nYou can now start the application:")
    logger.info("1. Start the backend:")
    logger.info("   cd backend")
    logger.info("   uvicorn app.main:app --reload")
    logger.info("\n2. In a new terminal, start the frontend:")
    logger.info("   cd dashboard")
    logger.info("   streamlit run main.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
