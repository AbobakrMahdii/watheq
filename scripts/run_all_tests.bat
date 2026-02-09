@echo off
REM Watheq - Run All Tests Script
REM This script runs all test suites across the entire project

echo ========================================
echo Watheq - Comprehensive Test Suite
echo ========================================
echo.

REM Check if in correct directory
if not exist "requirements.txt" (
    echo Error: Please run this script from the project root directory
    exit /b 1
)

echo [1/5] Running Backend API Tests...
echo ========================================
pip install -q pytest pytest-asyncio pytest-cov pytest-mock httpx faker 2>nul
pytest api/tests -v --cov=api --cov-report=html --cov-report=term
if errorlevel 1 (
    echo Backend tests FAILED
    set BACKEND_FAILED=1
) else (
    echo Backend tests PASSED
)
echo.

echo [2/5] Running AI/ML Tests...
echo ========================================
pytest ai/tests -v --cov=ai --cov-report=html --cov-report=term
if errorlevel 1 (
    echo AI tests FAILED
    set AI_FAILED=1
) else (
    echo AI tests PASSED
)
echo.

echo [3/5] Running Dashboard Unit Tests...
echo ========================================
cd dashboard
call npm install >nul 2>&1
call npm test -- --passWithNoTests
if errorlevel 1 (
    echo Dashboard unit tests FAILED
    set DASHBOARD_FAILED=1
) else (
    echo Dashboard unit tests PASSED
)
cd ..
echo.

echo [4/5] Running Mobile Tests...
echo ========================================
cd app
call flutter test
if errorlevel 1 (
    echo Mobile tests FAILED
    set MOBILE_FAILED=1
) else (
    echo Mobile tests PASSED
)
cd ..
echo.

echo [5/5] Test Summary
echo ========================================
if defined BACKEND_FAILED (echo Backend API Tests: FAILED) else (echo Backend API Tests: PASSED)
if defined AI_FAILED (echo AI/ML Tests: FAILED) else (echo AI/ML Tests: PASSED)
if defined DASHBOARD_FAILED (echo Dashboard Tests: FAILED) else (echo Dashboard Tests: PASSED)
if defined MOBILE_FAILED (echo Mobile Tests: FAILED) else (echo Mobile Tests: PASSED)
echo ========================================
echo.

if defined BACKEND_FAILED goto :failed
if defined AI_FAILED goto :failed
if defined DASHBOARD_FAILED goto :failed
if defined MOBILE_FAILED goto :failed

echo All tests PASSED! ✓
echo Coverage reports generated in htmlcov/ directories
exit /b 0

:failed
echo.
echo Some tests FAILED! ✗
echo Please review the output above for details
exit /b 1
