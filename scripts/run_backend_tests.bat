@echo off
REM Run Backend API Tests with Coverage

echo Running Backend API Tests...
cd /d %~dp0\..

REM Activate virtual environment
call .venv\Scripts\activate.bat 2>nul || (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
)

REM Install test dependencies if needed
pip install -q pytest pytest-asyncio pytest-cov pytest-mock httpx faker

REM Run tests
pytest api/tests -v --cov=api --cov-report=html --cov-report=term-missing

if errorlevel 1 (
    echo.
    echo Backend tests FAILED
    exit /b 1
) else (
    echo.
    echo Backend tests PASSED
    echo Coverage report: htmlcov/index.html
    exit /b 0
)
