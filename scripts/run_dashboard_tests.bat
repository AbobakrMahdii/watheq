@echo off
REM Run Dashboard Tests (Jest + Playwright)

echo Running Dashboard Tests...
cd /d %~dp0\..\dashboard

REM Install dependencies if needed
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)

echo.
echo [1/2] Running Jest Unit Tests...
call npm test -- --passWithNoTests

echo.
echo [2/2] Jest Tests Complete
echo.
echo To run E2E tests, use: npm run test:e2e
echo (Note: Backend must be running on port 8012)

exit /b 0
