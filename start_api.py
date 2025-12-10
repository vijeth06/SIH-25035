"""
Start API Server - Robust version
"""
import sys
import uvicorn
from simple_api import app

if __name__ == "__main__":
    print("🚀 Starting E-Consultation API on port 8002...")
    try:
        uvicorn.run(
            "simple_api:app",
            host="0.0.0.0",
            port=8002,
            log_level="info",
            access_log=True,
            reload=False
        )
    except KeyboardInterrupt:
        print("API server stopped by user")
    except Exception as e:
        print(f"Error starting API server: {e}")
        sys.exit(1)