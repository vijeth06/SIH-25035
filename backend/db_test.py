import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

# Load environment variables first
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Import database module
from app.core.database import MongoDB

# Check what the settings contain
from app.core.config import settings
print("Settings MONGODB_URI:", settings.MONGODB_URI)
print("Settings MONGODB_DB:", settings.MONGODB_DB)

# Try to connect to the database
import asyncio

async def test_db_connection():
    try:
        print("Connecting to database...")
        db = await MongoDB.connect_db()
        print("Connected successfully!")
        print("Database name:", db.name)
    except Exception as e:
        print("Connection failed:", e)

asyncio.run(test_db_connection())