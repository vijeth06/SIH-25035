import os
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pprint import pprint
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import settings
from app.core.config import settings

class MongoDBInitializer:
    def __init__(self, connection_string: str, db_name: str):
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None

    def connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.connection_string)
            # Test the connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info("Successfully connected to MongoDB!")
            return True
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False

    def initialize_collections(self):
        """Initialize required collections with validation and indexes."""
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Collections to create
        collections = {
            "users": {
                "validation": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["email", "hashed_password", "is_active", "created_at"],
                        "properties": {
                            "email": {"bsonType": "string", "description": "must be a string and is required"},
                            "hashed_password": {"bsonType": "string", "description": "must be a string and is required"},
                            "is_active": {"bsonType": "bool", "description": "must be a boolean and is required"},
                            "created_at": {"bsonType": "date", "description": "must be a date and is required"}
                        }
                    }
                },
                "indexes": [
                    [("email", 1)],  # Single field index on email
                ]
            },
            "analyses": {
                "validation": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["user_id", "text", "sentiment", "created_at"],
                        "properties": {
                            "user_id": {"bsonType": "objectId", "description": "must be an objectId and is required"},
                            "text": {"bsonType": "string", "description": "must be a string and is required"},
                            "sentiment": {
                                "bsonType": "object",
                                "required": ["label", "score"],
                                "properties": {
                                    "label": {"bsonType": "string"},
                                    "score": {"bsonType": "double"}
                                }
                            },
                            "created_at": {"bsonType": "date"}
                        }
                    }
                },
                "indexes": [
                    [("user_id", 1)],  # Index for user queries
                    [("created_at", -1)],  # Index for sorting by creation date
                ]
            },
            "comments": {
                "validation": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["text", "sentiment", "created_at"],
                        "properties": {
                            "text": {"bsonType": "string"},
                            "sentiment": {
                                "bsonType": "object",
                                "properties": {
                                    "label": {"bsonType": "string"},
                                    "score": {"bsonType": "double"}
                                }
                            },
                            "created_at": {"bsonType": "date"}
                        }
                    }
                },
                "indexes": [
                    [("created_at", -1)],
                    [("sentiment.label", 1)]
                ]
            },
            "reports": {
                "validation": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["name", "type", "created_at", "data"],
                        "properties": {
                            "name": {"bsonType": "string"},
                            "type": {"bsonType": "string"},
                            "created_at": {"bsonType": "date"},
                            "data": {"bsonType": "object"}
                        }
                    }
                },
                "indexes": [
                    [("created_at", -1)],
                    [("type", 1)]
                ]
            }
        }

        # Create collections with validation
        for collection_name, config in collections.items():
            try:
                # Create collection if it doesn't exist
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)
                    logger.info(f"Created collection: {collection_name}")
                
                # Update validation
                self.db.command({
                    "collMod": collection_name,
                    "validator": config["validation"],
                    "validationLevel": "strict",
                    "validationAction": "error"
                })
                
                # Create indexes
                for index_fields in config.get("indexes", []):
                    self.db[collection_name].create_index(index_fields)
                
                logger.info(f"Configured collection: {collection_name}")
                
            except OperationFailure as e:
                logger.warning(f"Could not configure collection {collection_name}: {e}")

    def create_sample_data(self):
        """Insert sample data for testing."""
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Sample users
        users = self.db.users
        if users.count_documents({}) == 0:
            sample_user = {
                "email": "admin@example.com",
                "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
                "is_active": True,
                "is_superuser": True,
                "created_at": datetime.utcnow()
            }
            users.insert_one(sample_user)
            logger.info("Created sample admin user (email: admin@example.com, password: secret)")

        # Sample comments
        comments = self.db.comments
        if comments.count_documents({}) == 0:
            sample_comments = [
                {
                    "text": "This is a positive comment about the service.",
                    "sentiment": {"label": "positive", "score": 0.95},
                    "created_at": datetime.utcnow()
                },
                {
                    "text": "I'm not very happy with the recent changes.",
                    "sentiment": {"label": "negative", "score": 0.75},
                    "created_at": datetime.utcnow()
                },
                {
                    "text": "The product is okay, but could be better.",
                    "sentiment": {"label": "neutral", "score": 0.5},
                    "created_at": datetime.utcnow()
                }
            ]
            comments.insert_many(sample_comments)
            logger.info("Inserted sample comments")

def main():
    # Get connection details from environment
    connection_string = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "sentiment_analysis")
    
    if not connection_string:
        logger.error("MONGODB_URI environment variable not set")
        sys.exit(1)
    
    # Initialize MongoDB
    initializer = MongoDBInitializer(connection_string, db_name)
    
    # Connect to MongoDB
    if not initializer.connect():
        sys.exit(1)
    
    # Initialize collections and indexes
    try:
        initializer.initialize_collections()
        initializer.create_sample_data()
        logger.info("MongoDB initialization completed successfully!")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
