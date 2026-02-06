@echo off
REM Check if Backend is running on the expected port (8012)
echo ========================================
echo Checking Backend API Status...
echo ========================================
echo.

netstat -ano | findstr :8012 >nul 2>&1
if errorlevel 1 (
    echo Backend is NOT running on port 8012
    echo.
    echo Start it with: start_backend.bat
) else (
    echo Backend appears to be running on port 8012
    echo.
    echo Testing connection...
    curl -s http://localhost:8012/api/v1/docs >nul 2>&1
    if errorlevel 1 (
        echo Port is open but API might not be responding.
        echo Try opening: http://localhost:8012/api/v1/docs
    ) else (
        echo Backend API is responding!
        echo Open: http://localhost:8012/api/v1/docs
    )
)

echo.
echo ========================================
pause
