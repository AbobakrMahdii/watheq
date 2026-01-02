@echo off
REM run_services.bat
REM Opens separate cmd windows for IPFS, Blockchain (if available), and API

setlocal

REM Root directory of repository (folder containing this script)
set ROOT_DIR=%~dp0

echo Repository root: %ROOT_DIR%

REM 1) Start IPFS using docker-compose (keeps window open to show logs/errors)
start "IPFS" cmd /k "cd /d "%ROOT_DIR%" && echo Starting IPFS via docker-compose... && docker-compose -f infrastructure\docker-compose.ledger.yml up --remove-orphans"

REM 2) Start Blockchain network
where wsl >nul 2>&1
if %ERRORLEVEL%==0 (
    start "Blockchain" cmd /k "cd /d "%ROOT_DIR%" && echo Starting blockchain via WSL... && wsl bash -lc 'cd /mnt/c/Users/sadeq/Desktop/watheq/ledger && ./scripts/start_network.sh' || echo Blockchain start failed; pause"
) else (
    start "Blockchain" cmd /k "cd /d "%ROOT_DIR%ledger" && echo No WSL found. To start the blockchain network run this in Git Bash or WSL: && echo ./scripts/start_network.sh && pause"
)

REM 3) Start API (keeps window open to show logs/errors)
timeout /t 5 /nobreak >nul
start "API" cmd /k "cd /d "%ROOT_DIR%" && echo Starting API (python -u -m api.main)... && python -u -m api.main"

endlocal

echo All windows started. Check the separate terminals for logs/errors.
