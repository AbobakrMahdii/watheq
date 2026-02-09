@echo off
REM Watheq AI Training Pipeline v3
REM Trains ElementClassifier + FontAnalyzer per document type

echo ========================================
echo   Watheq AI Training Pipeline v3
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
echo Uses layout_config.yaml per doc type.
echo.

REM Run the training script (--force to retrain even if already done)
python ai\train_ai.py --all --force

echo.
echo ========================================
echo AI Training Complete
echo ========================================
echo.
echo Press any key to close...
pause >nul
