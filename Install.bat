@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Pico-Algae Counter - Installer
echo ============================================================
echo.
echo  This sets everything up automatically:
echo    - installs Python if it is missing,
echo    - detects your NVIDIA GPU (or uses the CPU),
echo    - downloads the matching PyTorch and the components.
echo.
echo  You need an internet connection for this step. The first
echo  time it can take several minutes (PyTorch is a large download).
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup_env.ps1" -AppDir "%~dp0."
if errorlevel 1 (
    echo.
    echo   Setup did not finish. Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Double-click 'Start Pico-Algae Counter.bat' to run.
echo ============================================================
echo.
pause
