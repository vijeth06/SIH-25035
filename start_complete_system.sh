#!/bin/bash
# COMPLETE STARTUP SCRIPT - 200% Working System
# This script starts all services for the MCA Sentiment Analysis System

echo "🏛️ =========================================="
echo "🏛️  MCA SENTIMENT ANALYSIS SYSTEM STARTUP"
echo "🏛️ =========================================="
echo ""

# Function to check if port is in use
check_port() {
    local port=$1
    if netstat -an | grep ":$port " | grep LISTEN > /dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    echo "🔄 Checking port $port..."
    
    # For Windows PowerShell
    if command -v powershell > /dev/null 2>&1; then
        echo "🔍 Finding processes on port $port..."
        powershell -Command "Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id \$_ -Force -ErrorAction SilentlyContinue }"
        sleep 2
    # For Unix-like systems
    elif command -v lsof > /dev/null 2>&1; then
        echo "🔍 Finding processes on port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    echo "✅ Port $port cleared"
}

# Clear any existing processes on our ports
echo "🧹 Clearing existing processes..."
kill_port 8001  # API port
kill_port 8501  # Dashboard port

echo ""
echo "📁 Setting up directory structure..."

# Ensure we're in the right directory
if [ ! -f "backend/final_api.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Make sure backend/final_api.py exists"
    exit 1
fi

# Create necessary directories
mkdir -p data/sample
mkdir -p data/uploads
mkdir -p logs
mkdir -p backend/__pycache__
mkdir -p dashboard/__pycache__

echo "✅ Directory structure ready"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet fastapi uvicorn pandas numpy requests plotly streamlit python-multipart aiofiles python-dotenv

echo "✅ Dependencies installed"
echo ""

# Create sample data if it doesn't exist
if [ ! -f "data/sample/mca_test_dataset.csv" ]; then
    echo "📊 Creating sample MCA dataset..."
    python -c "
import pandas as pd
import os

# Create sample MCA consultation data
sample_data = [
    {'stakeholder_type': 'Individual', 'policy_area': 'Digital Governance', 'comment': 'I strongly support the new digital governance framework. It will improve transparency.'},
    {'stakeholder_type': 'NGO', 'policy_area': 'Corporate Affairs', 'comment': 'The proposed changes are excellent and will benefit small businesses significantly.'},
    {'stakeholder_type': 'Corporation', 'policy_area': 'Environmental Compliance', 'comment': 'These regulations are too strict and will harm business growth.'},
    {'stakeholder_type': 'Academic', 'policy_area': 'Digital Policy', 'comment': 'The framework needs more research-based evidence for effective implementation.'},
    {'stakeholder_type': 'Individual', 'policy_area': 'Corporate Affairs', 'comment': 'यह नई नीति बहुत अच्छी है और व्यापार को बढ़ावा देगी।'},
    {'stakeholder_type': 'Industry Association', 'policy_area': 'Digital Governance', 'comment': 'We appreciate the government initiative but request more consultation time.'},
    {'stakeholder_type': 'Legal Expert', 'policy_area': 'Environmental Compliance', 'comment': 'The legal framework is robust and addresses key environmental concerns effectively.'},
    {'stakeholder_type': 'Citizen', 'policy_area': 'Digital Policy', 'comment': 'This policy will make government services more accessible to common people.'},
    {'stakeholder_type': 'NGO', 'policy_area': 'Corporate Affairs', 'comment': 'নতুন নীতি অসাধারণ এবং এটি ছোট ব্যবসার জন্য খুবই উপকারী।'},
    {'stakeholder_type': 'Corporation', 'policy_area': 'Digital Governance', 'comment': 'The implementation timeline is unrealistic for large corporations.'}
]

# Create DataFrame and save
df = pd.DataFrame(sample_data)
os.makedirs('data/sample', exist_ok=True)
df.to_csv('data/sample/mca_test_dataset.csv', index=False)
print('✅ Sample dataset created: data/sample/mca_test_dataset.csv')
"
fi

echo ""
echo "🚀 Starting API Service..."
echo "📍 API will be available at: http://localhost:8001"
echo ""

# Start API in background
cd backend
python final_api.py &
API_PID=$!
cd ..

echo "⏳ Waiting for API to start..."
sleep 5

# Check if API is running
API_RUNNING=false
for i in {1..10}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        API_RUNNING=true
        echo "✅ API Service is running successfully!"
        break
    else
        echo "⏳ Checking API status... (attempt $i/10)"
        sleep 2
    fi
done

if [ "$API_RUNNING" = false ]; then
    echo "❌ API failed to start. Please check for errors."
    kill $API_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "🌐 Starting Dashboard..."
echo "📍 Dashboard will be available at: http://localhost:8501"
echo ""

# Start Dashboard
streamlit run dashboard/main.py --server.port 8501 --server.address 0.0.0.0 &
DASHBOARD_PID=$!

echo "⏳ Waiting for Dashboard to start..."
sleep 8

# Check if Dashboard is running
DASHBOARD_RUNNING=false
for i in {1..10}; do
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        DASHBOARD_RUNNING=true
        echo "✅ Dashboard is running successfully!"
        break
    else
        echo "⏳ Checking Dashboard status... (attempt $i/10)"
        sleep 2
    fi
done

echo ""
echo "🎉 =========================================="
echo "🎉  SYSTEM STARTUP COMPLETE!"
echo "🎉 =========================================="
echo ""
echo "📊 Services Running:"
echo "   🔹 API Service: http://localhost:8001"
echo "   🔹 Dashboard: http://localhost:8501"
echo "   🔹 Health Check: http://localhost:8001/health"
echo ""
echo "🚀 READY TO USE:"
echo "   1. Open: http://localhost:8501"
echo "   2. Click 'Load Sample Data'"
echo "   3. Explore the sentiment analysis results"
echo ""
echo "📋 Available Features:"
echo "   ✅ Multilingual sentiment analysis (15+ languages)"
echo "   ✅ Government-style MCA dashboard"
echo "   ✅ Real-time text analysis"
echo "   ✅ Word cloud generation"
echo "   ✅ CSV/JSON export"
echo "   ✅ Interactive visualizations"
echo ""
echo "🔧 Manual Commands (if needed):"
echo "   • API only: cd backend && python final_api.py"
echo "   • Dashboard only: streamlit run dashboard/main.py"
echo ""
echo "⏹️  To stop services: Press Ctrl+C or run: pkill -f 'python.*final_api' && pkill -f 'streamlit'"
echo ""

# Save PIDs for cleanup
echo $API_PID > .api_pid
echo $DASHBOARD_PID > .dashboard_pid

# Wait for user interruption
echo "🎯 System is running... Press Ctrl+C to stop all services"
trap 'echo -e "\n🛑 Stopping services..."; kill $API_PID $DASHBOARD_PID 2>/dev/null || true; rm -f .api_pid .dashboard_pid; echo "✅ All services stopped"; exit 0' INT

# Keep script running
wait