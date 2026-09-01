<#
  Pico-Algae Counter - environment setup.

  Does the whole "make it runnable" job, with no questions asked:
    1. Finds a suitable Python (>= 3.10); downloads and silently installs the
       official Python if none is present.
    2. Creates the .venv next to the app.
    3. Auto-detects the NVIDIA GPU / CUDA version (tools\detect_cuda.py) and
       installs the matching PyTorch build, or the CPU build if there is no GPU.
    4. Installs the rest of requirements.txt.

  Used by the Setup.exe wizard and by Install.bat. Safe to re-run (repair).

  Params:
    -AppDir   folder that contains app\, requirements.txt, tools\  (default: script's parent)
    -DryRun   do everything except the pip installs (for testing)
#>
param(
    [string]$AppDir = (Split-Path -Parent $PSScriptRoot),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Info($m) { Write-Host "  $m" }

function Find-Python {
    # Return path to a python.exe (>= 3.10), skipping the Store alias stub, or $null.
    $cands = @()
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $p = & py -3 -c "import sys;print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { $cands += $p.Trim() }
        } catch {}
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python -and $python.Source -and ($python.Source -notlike '*WindowsApps*')) {
        $cands += $python.Source
    }
    foreach ($c in ($cands | Select-Object -Unique)) {
        try {
            $v = & $c -c "import sys;print(sys.version_info[0]*100+sys.version_info[1])" 2>$null
            if ($LASTEXITCODE -eq 0 -and [int]$v -ge 310) { return $c }
        } catch {}
    }
    return $null
}

function Install-Python {
    $ver = '3.13.1'
    $url = "https://www.python.org/ftp/python/$ver/python-$ver-amd64.exe"
    $exe = Join-Path $env:TEMP "python-$ver-amd64.exe"
    Info "No suitable Python found - downloading Python $ver..."
    Invoke-WebRequest -Uri $url -OutFile $exe -UseBasicParsing
    Info "Installing Python (per-user, added to PATH)..."
    $args = '/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0 Include_pip=1'
    $p = Start-Process -FilePath $exe -ArgumentList $args -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Python installer exited with code $($p.ExitCode)." }
    Remove-Item $exe -Force -ErrorAction SilentlyContinue
    # Refresh PATH for this process so the new python is visible.
    $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path','User')
}

Write-Host "=================================================="
Write-Host "   Setting up Pico-Algae Counter"
Write-Host "=================================================="

$AppDir = (Resolve-Path $AppDir).Path
Info "Install folder: $AppDir"

$py = Find-Python
if (-not $py) {
    Install-Python
    $py = Find-Python
    if (-not $py) { throw "Python installation did not complete. Please install Python 3.13 from python.org and run again." }
}
Info "Using Python: $py"

$venv = Join-Path $AppDir '.venv'
$venvPy = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    Info "Creating virtual environment..."
    & $py -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}

# Guarantee pip exists in the venv. A venv created by `uv venv` ships without pip,
# so a reused .venv can be pip-less -- bootstrap it rather than failing later.
& $venvPy -m pip --version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Info "Bootstrapping pip in the virtual environment..."
    & $venvPy -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw "Could not bootstrap pip in the virtual environment." }
}

$channel = (& $py (Join-Path $AppDir 'tools\detect_cuda.py')).Trim()
if (-not $channel) { $channel = 'cpu' }
if ($channel -eq 'cpu') {
    Info "No NVIDIA GPU detected -> installing the CPU build of PyTorch."
} else {
    Info "NVIDIA GPU detected -> installing the GPU build of PyTorch ($channel)."
}

if ($DryRun) {
    Info "[DryRun] Would install torch/torchvision from channel '$channel' and requirements.txt."
    Write-Host "Dry run OK."
    return
}

Info "Upgrading pip..."
& $venvPy -m pip install --upgrade pip --quiet

Info "Installing PyTorch ($channel) - this can take a few minutes the first time..."
& $venvPy -m pip install torch==2.10.0 torchvision==0.25.0 --index-url "https://download.pytorch.org/whl/$channel"
if ($LASTEXITCODE -ne 0) { throw "Installing PyTorch failed. Usually the internet connection; if you are online, the .venv may be broken -- delete the .venv folder and re-run." }

Info "Installing the remaining components..."
& $venvPy -m pip install -r (Join-Path $AppDir 'requirements.txt') --quiet
if ($LASTEXITCODE -ne 0) { throw "Installing requirements failed." }

Write-Host ""
Write-Host "Setup complete."
