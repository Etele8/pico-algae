<#
  Build the Pico-Algae Counter Setup.exe.

  Stamps app\VERSION.txt from the current git commit (so the in-app update
  check has a baseline), then compiles installer\pico.iss with Inno Setup.

  IMPORTANT: build from a commit that is already pushed to GitHub main, or the
  in-app "update available" check will compare against an older remote commit.

  Usage (from the repo root):   powershell -ExecutionPolicy Bypass -File installer\build.ps1
#>
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

# 1) Stamp the packaged version from git HEAD (short sha + author date).
$sha  = (& git rev-parse --short HEAD).Trim()
$date = (& git show -s --format=%cI HEAD).Trim()
"$sha  $date" | Out-File -Encoding ascii (Join-Path $root 'app\VERSION.txt')
Write-Host "Stamped app\VERSION.txt -> $sha  $date"

$unpushed = (& git rev-list "origin/main..HEAD" --count).Trim()
if ($unpushed -ne '0') {
    Write-Warning "HEAD is $unpushed commit(s) ahead of origin/main. Push before distributing, or the update check will be wrong."
}

# 2) Locate the Inno Setup compiler.
$iscc = @(
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "ISCC.exe (Inno Setup 6) not found. Install it: winget install JRSoftware.InnoSetup" }

# 3) Compile.
& $iscc (Join-Path $root 'installer\pico.iss')
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed ($LASTEXITCODE)." }
Write-Host ""
Write-Host "Built: $root\Pico-Algae-Counter-Setup.exe"
