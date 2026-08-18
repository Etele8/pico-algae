@echo off
setlocal EnableExtensions
title Pico-Algae Counter - Update
cd /d "%~dp0"

echo ==================================================
echo    Updating Pico-Algae Counter
echo ==================================================
echo.
echo This downloads the newest version of the program from
echo the internet. Your saved images, your settings and the
echo counting model are NOT changed.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; try { [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $url='https://github.com/Etele8/pico-algae/archive/refs/heads/main.zip'; $tmp=Join-Path $env:TEMP ('pico_upd_'+[guid]::NewGuid().ToString('N')); New-Item -ItemType Directory -Path $tmp | Out-Null; $zip=Join-Path $tmp 'src.zip'; Write-Host '  Downloading...'; Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing; Write-Host '  Extracting...'; Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory($zip,$tmp); $r=Join-Path $tmp 'pico-algae-main'; Write-Host '  Installing files...'; Copy-Item (Join-Path $r 'Install.bat') '.' -Force; Copy-Item (Join-Path $r 'Start Pico-Algae Counter.bat') '.' -Force; Copy-Item (Join-Path $r 'Capture from Olympus.bat') '.' -Force; Copy-Item (Join-Path $r 'requirements.txt') '.' -Force; if(!(Test-Path 'app')){ New-Item -ItemType Directory 'app' | Out-Null }; Copy-Item (Join-Path $r 'app\*') 'app' -Recurse -Force; try { $c=Invoke-RestMethod -Uri 'https://api.github.com/repos/Etele8/pico-algae/commits/main' -Headers @{'User-Agent'='pico-updater'}; ($c.sha.Substring(0,7)+'  '+$c.commit.author.date) | Out-File -Encoding ascii 'app\VERSION.txt' } catch {}; Remove-Item $tmp -Recurse -Force; Write-Host '  OK'; } catch { Write-Host ''; Write-Host ('  UPDATE FAILED: '+$_.Exception.Message); exit 1 }"

if errorlevel 1 goto failed

echo.
echo Refreshing components (quick)...
if exist ".venv\Scripts\python.exe" ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt

echo.
echo ==================================================
echo    Update complete.
echo ==================================================
if exist "app\VERSION.txt" (
  echo Installed version:
  type "app\VERSION.txt"
)
echo.
echo Close this window, then start the program as usual.
echo.
pause
exit /b 0

:failed
echo.
echo Could not update. Check the internet connection and
echo try again. Your current version still works fine.
echo.
pause
exit /b 1
