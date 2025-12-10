import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

# Load environment variables first
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

print("Before importing settings:")
import os
print("MONGODB_URI in os.environ:", "MONGODB_URI" in os.environ)
if "MONGODB_URI" in os.environ:
    print("MONGODB_URI value:", os.environ["MONGODB_URI"])

# Import settings
from app.core.config import settings

print("After importing settings:")
print("MONGODB_URI in os.environ:", "MONGODB_URI" in os.environ)
print("MONGODB_URI from settings:", settings.MONGODB_URI)