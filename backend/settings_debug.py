import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

# Load environment variables first
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import os
print("Environment variables after loading .env:")
print("MONGODB_URI:", os.environ.get("MONGODB_URI"))
print("MONGODB_DB:", os.environ.get("MONGODB_DB"))

# Import settings
from app.core.config import settings

print("\nSettings after importing:")
print("MONGODB_URI:", settings.MONGODB_URI)
print("MONGODB_DB:", settings.MONGODB_DB)

# Import database module
from app.core.database import settings as db_settings

print("\nDatabase module settings:")
print("MONGODB_URI:", db_settings.MONGODB_URI)
print("MONGODB_DB:", db_settings.MONGODB_DB)