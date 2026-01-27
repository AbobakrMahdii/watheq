@echo off
REM Check if Backend is running
echo ========================================
echo Checking Backend API Status...
echo ========================================
echo.

REM Check if port 8001 is listening
netstat -ano | findstr :8001 >nul 2>&1
if errorlevel 1 (
    echo ❌ Backend is NOT running on port 8001
    echo.
    echo Please start Backend using: start_backend.bat
) else (
    echo ✅ Backend appears to be running on port 8001
    echo.
    echo Testing connection...
    curl -s http://localhost:8001/api/v1/docs >nul 2>&1
    if errorlevel 1 (
        echo ⚠️  Port is open but API might not be responding
        echo    Try opening: http://localhost:8001/api/v1/docs
    ) else (
        echo ✅ Backend API is responding!
        echo    Open: http://localhost:8001/api/v1/docs
    )
)

echo.
echo ========================================
pause
