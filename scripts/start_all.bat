@echo off
REM Start all services for Watheq project
echo ========================================
echo Starting All Watheq Services
echo ========================================
echo.

cd /d "%~dp0.."

REM Check if PyTorch is installed (required for AI training)
echo [0/5] Checking AI models (v3 — ElementClassifier + FontAnalyzer)...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo      WARNING: PyTorch not installed. AI training skipped.
    echo      Run:  scripts\setup_python.bat  to install all dependencies.
    goto :skip_ai
)
python ai\train_ai.py --all
if errorlevel 0 (
    echo      AI models ready.
) else (
    echo      Warning: AI training check failed. Continuing anyway...
)
:skip_ai
echo.

REM Start IPFS in a new window (Docker)
echo [1/5] Starting IPFS...
start "Watheq - IPFS" cmd /c "%~dp0start_ipfs.bat"

REM Wait a bit for IPFS to start
timeout /t 3 /nobreak >nul

REM Start MultiChain in a new window (Docker)
echo [2/5] Starting MultiChain Blockchain...
start "Watheq - MultiChain" cmd /c "%~dp0start_multichain.bat"

REM Wait a bit for MultiChain RPC to become ready
timeout /t 5 /nobreak >nul

REM Start Backend API in a new window
echo [3/5] Starting Backend API...
start "Watheq - Backend API" cmd /k "%~dp0start_backend.bat"

REM Wait a bit for Backend to start
timeout /t 4 /nobreak >nul

REM Start Dashboard in a new window
echo [4/5] Starting Dashboard...
start "Watheq - Dashboard" cmd /k "%~dp0start_dashboard.bat"

echo.
echo ========================================
echo All services are starting in separate windows
echo ========================================
echo.
echo Services:
echo   - IPFS:        http://localhost:15001 (API), http://localhost:18080 (Gateway)
echo   - MultiChain:  http://localhost:4402 (JSON-RPC) - watheqchain
echo   - Backend API: http://localhost:8012
echo   - API Docs:    http://localhost:8012/api/v1/docs
echo   - Dashboard:   http://localhost:3000
echo.
timeout /t 5 /nobreak >nul
