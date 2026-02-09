@echo off
REM Run Flutter Mobile Tests

echo Running Flutter Mobile Tests...
cd /d %~dp0\..\app

REM Run all Flutter tests
flutter test

if errorlevel 1 (
    echo.
    echo Mobile tests FAILED
    exit /b 1
) else (
    echo.
    echo Mobile tests PASSED
    echo.
    echo To run with coverage: flutter test --coverage
    exit /b 0
)
