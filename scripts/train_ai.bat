@echo off
REM Train AI Models for Watheq
REM Runs training for all document types that haven't been trained yet

echo ========================================
echo Watheq AI Training
echo ========================================
echo.

cd /d "%~dp0.."

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting AI training for all document types...
echo This will skip already trained types.
echo.

REM Run the training script
python ai\train_ai.py --all

echo.
echo ========================================
echo AI Training Complete
echo ========================================
echo.
echo Press any key to close...
pause >nul
