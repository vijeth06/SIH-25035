#!/bin/bash

echo "============================================"
echo " E-Consultation Platform - Quick Start"
echo "============================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

echo "[1/3] Starting Backend API (Port 8001)..."
cd backend
uvicorn app.main:app --reload --port 8001 --host 0.0.0.0 &
BACKEND_PID=$!
cd ..

echo "[2/3] Waiting for backend to initialize..."
sleep 5

echo "[3/3] Starting Streamlit Dashboard (Port 8501)..."
streamlit run dashboard/main.py --server.port 8501 &
DASHBOARD_PID=$!

echo ""
echo "============================================"
echo " All Services Started Successfully!"
echo "============================================"
echo ""
echo "  Backend API:  http://localhost:8001"
echo "  API Docs:     http://localhost:8001/docs"
echo "  Dashboard:    http://localhost:8501"
echo ""
echo "  Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "kill $BACKEND_PID $DASHBOARD_PID; exit" INT
wait
