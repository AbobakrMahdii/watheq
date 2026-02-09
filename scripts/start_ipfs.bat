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

REM Remove any stale container from previous runs
docker rm -f ipfs-node >nul 2>&1

echo Starting IPFS container...
docker-compose -p watheq-ipfs -f "%COMPOSE_FILE%" up -d --remove-orphans

REM Wait for IPFS API to become responsive
echo Waiting for IPFS API...
set /a TRIES=0
:wait_loop
timeout /t 2 /nobreak >nul
set /a TRIES+=1
curl -s -o nul -w "%%{http_code}" -X POST http://127.0.0.1:15001/api/v0/id 2>nul | findstr /C:"200" >nul 2>&1
if %ERRORLEVEL%==0 goto :ready
if %TRIES% GEQ 15 (
    echo WARNING: IPFS API did not respond in 30s. Check Docker logs:
    echo   docker logs ipfs-node
    goto :EOF
)
goto :wait_loop

:ready
echo IPFS is running and ready.
echo   API:      http://127.0.0.1:15001
echo   Gateway:  http://127.0.0.1:18080
echo.
timeout /t 3 /nobreak >nul
