@echo off
REM Start all services for Watheq project
echo ========================================
echo Starting All Watheq Services
echo ========================================
echo.

cd /d "%~dp0.."

REM Run AI training for any untrained document types (skips already trained)
echo [0/3] Checking AI models...
python ai\train_ai.py --all 2>nul
if errorlevel 0 (
    echo      AI models ready.
) else (
    echo      Warning: AI training check failed. Continuing anyway...
)
echo.

REM Start IPFS in a new window (optional; skips silently if compose file missing)
echo [1/3] Starting IPFS...
start "Watheq - IPFS" cmd /k "%~dp0start_ipfs.bat"

REM Wait a bit for IPFS to start
timeout /t 2 /nobreak >nul

REM Start Backend API in a new window
echo [2/3] Starting Backend API...
start "Watheq - Backend API" cmd /k "%~dp0start_backend.bat"

REM Wait a bit for Backend to start
timeout /t 4 /nobreak >nul

REM Start Dashboard in a new window
echo [3/3] Starting Dashboard...
start "Watheq - Dashboard" cmd /k "%~dp0start_dashboard.bat"

echo.
echo ========================================
echo All services are starting in separate windows
echo ========================================
echo.
echo Services:
echo   - IPFS:        http://localhost:15001 (API), http://localhost:18080 (Gateway) [optional if compose file exists]
echo   - Backend API: http://localhost:8012
echo   - API Docs:    http://localhost:8012/api/v1/docs
echo   - Dashboard:   http://localhost:3000
echo.
echo Press any key to close this window (services will keep running)...
pause >nul
