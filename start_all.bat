@echo off
REM Start all services for Watheq project
echo ========================================
echo Starting All Watheq Services
echo ========================================
echo.

cd /d "%~dp0"

REM Start IPFS in a new window
echo [1/3] Starting IPFS...
start "Watheq - IPFS" cmd /k "%~dp0start_ipfs.bat"

REM Wait a bit for IPFS to start
timeout /t 3 /nobreak >nul

REM Start Backend API in a new window
echo [2/3] Starting Backend API...
start "Watheq - Backend API" cmd /k "%~dp0start_backend.bat"

REM Wait a bit for Backend to start
timeout /t 5 /nobreak >nul

REM Start Dashboard in a new window
echo [3/3] Starting Dashboard...
start "Watheq - Dashboard" cmd /k "%~dp0start_dashboard.bat"

echo.
echo ========================================
echo All services are starting in separate windows
echo ========================================
echo.
echo Services:
echo   - IPFS:        http://localhost:5001 (API), http://localhost:8081 (Gateway)
echo   - Backend API: http://localhost:8001
echo   - API Docs:    http://localhost:8001/api/v1/docs
echo   - Dashboard:   http://localhost:3000
echo.
echo Press any key to close this window (services will keep running)...
pause >nul
