# E-Consultation Platform - PowerShell Launcher
# More robust than batch file with better error handling

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " E-Consultation Platform - Quick Start" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv .venv" -ForegroundColor Yellow
    Write-Host "Then: .venv\Scripts\activate" -ForegroundColor Yellow
    Write-Host "Then: pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Function to check if port is in use
function Test-Port {
    param($Port)
    $connection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
    return $connection.TcpTestSucceeded
}

# Check if ports are available
Write-Host "[Checking ports...]" -ForegroundColor Yellow
if (Test-Port 8001) {
    Write-Host "WARNING: Port 8001 is already in use!" -ForegroundColor Yellow
    $response = Read-Host "Do you want to kill the process? (Y/N)"
    if ($response -eq "Y" -or $response -eq "y") {
        $process = Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue
        if ($process) {
            Stop-Process -Id $process.OwningProcess -Force
            Write-Host "Process killed successfully" -ForegroundColor Green
        }
    }
}

if (Test-Port 8501) {
    Write-Host "WARNING: Port 8501 is already in use!" -ForegroundColor Yellow
    $response = Read-Host "Do you want to kill the process? (Y/N)"
    if ($response -eq "Y" -or $response -eq "y") {
        $process = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
        if ($process) {
            Stop-Process -Id $process.OwningProcess -Force
            Write-Host "Process killed successfully" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "[1/3] Starting Backend API (Port 8001)..." -ForegroundColor Green

# Start Backend in new window
$backendScript = @"
Set-Location '$PSScriptRoot\backend'
& '$PSScriptRoot\.venv\Scripts\Activate.ps1'
Write-Host 'Backend API Server Starting...' -ForegroundColor Cyan
uvicorn app.main:app --reload --port 8001 --host 0.0.0.0
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendScript

Write-Host "[2/3] Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Test backend health
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8001/api/v1/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Backend API is healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "! Backend may still be starting..." -ForegroundColor Yellow
}

Write-Host "[3/3] Starting Streamlit Dashboard (Port 8501)..." -ForegroundColor Green

# Start Dashboard in new window
$dashboardScript = @"
Set-Location '$PSScriptRoot'
& '$PSScriptRoot\.venv\Scripts\Activate.ps1'
Write-Host 'Streamlit Dashboard Starting...' -ForegroundColor Cyan
streamlit run dashboard\main.py --server.port 8501
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $dashboardScript

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " All Services Started Successfully!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backend API:  " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8001" -ForegroundColor Yellow
Write-Host "  API Docs:     " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8001/docs" -ForegroundColor Yellow
Write-Host "  Dashboard:    " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Opening dashboard in default browser..." -ForegroundColor Cyan
Start-Sleep -Seconds 3
Start-Process "http://localhost:8501"

Write-Host ""
Write-Host "  Both services are running in separate windows." -ForegroundColor White
Write-Host "  Close those windows to stop the services." -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit this launcher"
