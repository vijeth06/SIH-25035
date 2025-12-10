#!/bin/bash
# Comprehensive startup script for Government Policy Feedback Analysis Platform

echo "🛠️ Government Policy Feedback Analysis Platform Startup"
echo "======================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Running setup first..."
    if [ -f "setup_environment.bat" ]; then
        echo "Run setup_environment.bat first on Windows"
        exit 1
    else
        echo "Creating virtual environment..."
        python -m venv .venv
    fi
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate || .venv\Scripts\activate.bat

# Start backend
echo "🚀 Starting backend server..."
echo "Backend will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"

# Start backend in background
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Start frontend
echo "🌐 Starting frontend dashboard..."
echo "Dashboard will be available at: http://localhost:3000"

cd frontend
npm install --legacy-peer-deps
npm start &
FRONTEND_PID=$!

cd ..

echo "✅ Platform started successfully!"
echo ""
echo "🔗 Access URLs:"
echo "   Dashboard: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "📋 Default Login Credentials:"
echo "   Email: admin@gov.in"
echo "   Password: admin12345"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup INT TERM

# Wait for user to stop
wait