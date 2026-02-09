@echo off
REM Start MultiChain Service using Docker
echo ========================================
echo Starting MultiChain Blockchain...
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

REM Start MultiChain using docker-compose if compose file exists
set "COMPOSE_FILE=%~dp0..\infrastructure\docker-compose.multichain.yml"
if not exist "%COMPOSE_FILE%" (
    echo WARNING: %COMPOSE_FILE% not found. Skipping MultiChain startup.
    goto :EOF
)

REM Remove any stale container from previous runs
docker rm -f multichain-node >nul 2>&1

echo Starting MultiChain container...
docker-compose -p watheq-multichain -f "%COMPOSE_FILE%" up -d --build --remove-orphans

REM Wait for RPC to become responsive
echo Waiting for MultiChain RPC...
set /a TRIES=0
:wait_loop
timeout /t 2 /nobreak >nul
set /a TRIES+=1
curl -s -o nul -w "%%{http_code}" -u watheqrpc:watheqrpcpass --data-binary "{\"jsonrpc\":\"1.0\",\"id\":\"test\",\"method\":\"getinfo\",\"params\":[]}" -H "Content-Type: application/json" http://127.0.0.1:4402 | findstr /C:"200" >nul 2>&1
if %ERRORLEVEL%==0 goto :ready
if %TRIES% GEQ 20 (
    echo WARNING: MultiChain RPC did not respond in 40s. Check Docker logs:
    echo   docker logs multichain-node
    goto :EOF
)
goto :wait_loop

:ready
echo MultiChain is running and ready.
echo   Chain:    watheqchain
echo   RPC:      http://127.0.0.1:4402
echo   Stream:   documents (auto-created)
echo.
timeout /t 3 /nobreak >nul
