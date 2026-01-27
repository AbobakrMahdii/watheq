@echo off
REM Start IPFS Service using Docker
echo ========================================
echo Starting IPFS Service...
echo ========================================

cd /d "%~dp0"

REM Check if Docker is installed and running
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not running
    echo Please install Docker Desktop and start it
    pause
    exit /b 1
)

REM Start IPFS using docker-compose
echo Starting IPFS container...
docker-compose -f infrastructure\docker-compose.ledger.yml up --remove-orphans

pause
