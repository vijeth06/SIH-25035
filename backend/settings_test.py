import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

# Load environment variables first
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Import settings
from app.core.config import settings

print("Settings MONGODB_URI:", settings.MONGODB_URI)
print("Settings MONGODB_DB:", settings.MONGODB_DB)

# Import database module and check what it sees
from app.core.database import settings as db_settings

print("Database module settings MONGODB_URI:", db_settings.MONGODB_URI)
print("Database module settings MONGODB_DB:", db_settings.MONGODB_DB)