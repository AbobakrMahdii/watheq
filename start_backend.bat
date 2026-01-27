@echo off
REM Start Backend API Server
echo ========================================
echo Starting Watheq Backend API...
echo ========================================

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists, if not create it
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/Update dependencies
echo Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

REM Check if MySQL is running (optional check)
echo.
echo Checking database connection...
python -c "import mysql.connector; print('MySQL connector available')" 2>nul || echo Warning: MySQL connector not found. Make sure MySQL is running.

REM Start the API server
echo.
echo ========================================
echo Starting API server on http://localhost:8001
echo API Docs: http://localhost:8001/api/v1/docs
echo ========================================
echo.

python -u -m api.main

pause
