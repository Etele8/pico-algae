@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Pico-Algae Counter - Installer
echo ============================================================
echo.
echo  Choose how to run the model:
echo.
echo    [1] CPU only            Works on any PC. No graphics card needed.
echo    [2] NVIDIA GPU - CUDA 12.6   Recommended for most NVIDIA cards.
echo    [3] NVIDIA GPU - CUDA 12.8   For newer NVIDIA drivers.
echo    [4] NVIDIA GPU - CUDA 13.0   For the latest NVIDIA drivers.
echo.
echo  Pick a GPU option only if this PC has an NVIDIA graphics card.
echo  If a GPU option fails to detect the card later, re-run this and
echo  choose CPU, or try a different CUDA version.
echo.

set "TORCH_INDEX="
set /p choice="Enter 1, 2, 3 or 4: "
if "%choice%"=="1" set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
if "%choice%"=="2" set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
if "%choice%"=="3" set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
if "%choice%"=="4" set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"

if not defined TORCH_INDEX (
    echo.
    echo   Invalid choice "%choice%". Please run the installer again.
    echo.
    pause
    exit /b 1
)

echo.
echo  Installing from: %TORCH_INDEX%
echo.

REM Find an installed Python. Any of 3.10-3.14 works (3.13 recommended).
set "PY="
for %%V in (3.13 3.12 3.11 3.10 3.14) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
    )
)
if not defined PY ( where py >nul 2>nul && set "PY=py" )
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
    echo.
    echo   ERROR: No Python found. Install Python 3.13 from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH" during setup, then re-run this.
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%i in ('%PY% -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%i"
echo  Using Python %PYVER%  (%PY%)
echo.

if not exist ".venv\Scripts\python.exe" (
    echo  Creating the program environment ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo.
        echo   ERROR: Could not create the virtual environment.
        echo   Make sure Python 3.10 or newer is installed and on PATH.
        echo.
        pause
        exit /b 1
    )
)

set "VPY=.venv\Scripts\python.exe"

"%VPY%" -m pip install --upgrade pip
echo.
echo  Installing PyTorch (this is a large download, please wait) ...
"%VPY%" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo.
    echo   ERROR: PyTorch install failed. Check your internet connection,
    echo   or try a different CUDA version.
    echo.
    pause
    exit /b 1
)

echo.
echo  Installing the rest of the dependencies ...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo   ERROR: Dependency install failed.
    echo.
    pause
    exit /b 1
)

echo.
echo  Verifying installation ...
"%VPY%" -c "import torch; print('PyTorch', torch.__version__); print('GPU detected:', torch.cuda.is_available())"

echo.
echo ============================================================
echo   Setup complete.
echo   Double-click 'Start Pico-Algae Counter.bat' to run.
echo ============================================================
echo.
pause
