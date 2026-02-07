@echo off
REM Start IPFS Service using Docker
echo ========================================
echo Starting IPFS Service...
echo ========================================

cd /d "%~dp0.."

REM Check if Docker is installed and running
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not running
    echo Please install Docker Desktop and start it
    pause
    exit /b 1
)

REM Start IPFS using docker-compose if compose file exists
set "COMPOSE_FILE=%~dp0..\infrastructure\docker-compose.ipfs.yml"
if not exist "%COMPOSE_FILE%" (
    echo WARNING: %COMPOSE_FILE% not found. Skipping IPFS startup.
    goto :EOF
)

echo Starting IPFS container...
docker-compose -f "%COMPOSE_FILE%" up --remove-orphans

pause
