@echo off
REM Start Backend API Server
echo ========================================
echo Starting Watheq Backend API...
echo ========================================

cd /d "%~dp0.."

REM Use single project virtual environment in root
set "VENV_DIR=%~dp0../.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"
set "REQ_UNIFIED=%~dp0../requirements.unified.txt"
set "REQ_API=%~dp0../api\requirements.txt"

REM Check if Python 3.13 is available
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.13 is not installed.
    echo Please install Python 3.13 and try again.
    pause
    exit /b 1
)

REM Create .venv if it doesn't exist
if not exist "%PYTHON_EXE%" (
    echo Creating virtual environment .venv...
    py -3.13 -m venv "%VENV_DIR%"
)

REM Install dependencies only if needed
if not exist "%VENV_DIR%\Lib\site-packages\fastapi" (
    echo Installing dependencies - one-time...
    if exist "%REQ_UNIFIED%" (
        "%PIP_EXE%" install -r "%REQ_UNIFIED%"
    ) else (
        "%PIP_EXE%" install -r "%REQ_API%"
    )
)

REM Check if MySQL is running (optional check)
echo.
echo Checking database connection...
"%PYTHON_EXE%" -c "import aiomysql; print('MySQL driver available')" 2>nul || echo Warning: MySQL driver not found. Make sure MySQL is running.

REM Start the API server
echo.
echo ========================================
echo Starting API server on http://localhost:8012
echo API Docs: http://localhost:8012/api/v1/docs
echo ========================================
echo.

REM Start the API server using .venv
"%PYTHON_EXE%" -u -m api.main

pause
