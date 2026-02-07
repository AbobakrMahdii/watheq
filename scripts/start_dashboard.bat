@echo off
REM Start Frontend Dashboard
echo ========================================
echo Starting Watheq Admin Dashboard...
echo ========================================

cd /d "%~dp0..\dashboard"

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18+ and add it to PATH
    pause
    exit /b 1
)

REM Check if node_modules exists, if not install dependencies
if not exist "node_modules\" (
    echo Installing dependencies...
    call npm install
)

REM Check if .env.local exists, if not create from example
if not exist ".env.local" (
    echo Creating .env.local from example...
    copy .env.local.example .env.local >nul 2>&1
    echo Created .env.local - Please check BACKEND_BASE_URL if needed
)

REM Start the development server
echo.
echo ========================================
echo Starting Dashboard on http://localhost:3000
echo ========================================
echo.

call npm run dev

pause
