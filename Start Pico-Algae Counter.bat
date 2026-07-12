@echo off
REM ============================================================
REM  Pico-Algae Counter - double-click launcher
REM  Starts the local web app and opens it in your browser.
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
echo   Starting Pico-Algae Counter...
echo   A browser window will open automatically.
echo   Keep this window open while you use the program.
echo.

".venv\Scripts\python.exe" "app\server.py"

echo.
echo   The program has stopped.
pause
