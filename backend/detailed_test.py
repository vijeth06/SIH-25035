import os
from pathlib import Path
from dotenv import load_dotenv

print("Before loading .env:")
print("MONGODB_URI in os.environ:", "MONGODB_URI" in os.environ)
if "MONGODB_URI" in os.environ:
    print("MONGODB_URI value:", os.environ["MONGODB_URI"])

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
print("Env file path:", env_path)
print("Env file exists:", env_path.exists())

if env_path.exists():
    with open(env_path, 'r') as f:
        content = f.read()
        print("Env file content:")
        print(content)

# Load environment variables
load_dotenv(dotenv_path=env_path)

print("\nAfter loading .env:")
print("MONGODB_URI in os.environ:", "MONGODB_URI" in os.environ)
if "MONGODB_URI" in os.environ:
    print("MONGODB_URI value:", os.environ["MONGODB_URI"])

# Now import the settings
from app.core.config import settings

print("\nSettings values:")
print("MONGODB_URI:", settings.MONGODB_URI)
print("MONGODB_DB:", settings.MONGODB_DB)