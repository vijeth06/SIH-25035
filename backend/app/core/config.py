import os
import logging
from typing import List, Optional, Union, Dict, Any
from pydantic import AnyHttpUrl, EmailStr, validator
from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# The .env file is in the project root directory, not in the backend directory
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
print("DEBUG: Env file path:", env_path)
print("DEBUG: Env file exists:", env_path.exists())

# Load environment variables
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print("DEBUG: Loaded .env file")
else:
    print("DEBUG: .env file not found")

# Debug: Print environment variables after loading
print("DEBUG: Environment variables after loading .env:")
print("DEBUG: MONGODB_URI:", os.environ.get("MONGODB_URI"))
print("DEBUG: MONGODB_DB:", os.environ.get("MONGODB_DB"))

# Ensure environment variables are loaded before defining the Settings class
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = os.getenv("MONGODB_DB", "sentiment_analysis")

# Debug: Print the values being used
print("DEBUG: MONGODB_URI being used:", MONGODB_URI)
print("DEBUG: MONGODB_DB being used:", MONGODB_DB)

class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    APP_NAME: str = "E-Consultation Insight Engine"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"

    # Default language for multilingual processing
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # Frontend URL
        "http://localhost:8000",  # Backend URL
        "http://localhost:8501",  # Streamlit dashboard
    ]
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Security
    ALLOWED_HOSTS: List[str] = ["*"]  # In production, replace with your domain
    
    # MongoDB settings
    MONGODB_URI: str = MONGODB_URI
    MONGODB_DB: str = MONGODB_DB

    # File upload settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx", ".xls", ".txt", ".json"]
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))  # 50MB default
    
    # First superuser
    FIRST_SUPERUSER_EMAIL: EmailStr = os.getenv("FIRST_SUPERUSER_EMAIL", "admin@econsultation.gov")
    FIRST_SUPERUSER_PASSWORD: str = os.getenv("FIRST_SUPERUSER_PASSWORD", "admin123")
    
    # JWT settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    class Config:
        case_sensitive = True
        # Don't rely on pydantic-settings to load .env file since it's not working
        # We're explicitly loading it with load_dotenv above
        env_file = None
        extra = "ignore"  # Ignore extra environment variables without raising validation errors

# Create settings instance
settings = Settings()

# Debug: Print the settings values
print("DEBUG: Settings MONGODB_URI:", settings.MONGODB_URI)
print("DEBUG: Settings MONGODB_DB:", settings.MONGODB_DB)

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)