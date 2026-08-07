@echo off
REM ============================================================
REM  Pico-Algae Counter - Capture from Olympus
REM  Starts the app plus a floating "Capture" button so you can
REM  screenshot the microscopy image straight off the screen.
REM ============================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   ERROR: Python environment not found at .venv\Scripts\python.exe
    echo   Please run 'Install.bat' first to set up the program.
    echo.
    pause
    exit /b 1
)

echo.
echo   Starting Pico-Algae Capture...
echo   - A small "Pico Capture" window appears - keep it on screen.
echo   - Click "Select region" once to mark the Olympus image area.
echo   - Then "Capture and Review" screenshots it and opens the browser.
echo   Keep THIS window open while you work; close it to stop.
echo.

".venv\Scripts\python.exe" "app\capture_app.py"

echo.
echo   The program has stopped.
pause
