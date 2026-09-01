<#
  Fetch the counting model if a newer one has been released.

  Reads app\model.json ({version, url, sha256}) and compares it to the installed
  model's stamp (runs\tuning\train\MODEL_VERSION.txt). Downloads the model from
  the release URL only when it differs, verifies the SHA-256, and drops it at
  runs\tuning\train\best_train_model.pt.

  Called by Update.bat after the code is refreshed. Never aborts the update:
  any problem here just warns and leaves the current model in place (exit 0), so
  a model hiccup can't break a working install.

  Params:
    -AppDir   folder containing app\, runs\ (default: script's parent)
#>
param(
    [string]$AppDir = (Split-Path -Parent $PSScriptRoot)
)

$ProgressPreference = 'SilentlyContinue'
function Info($m) { Write-Host "  $m" }

try {
    $AppDir   = (Resolve-Path $AppDir).Path
    $manifest = Join-Path $AppDir 'app\model.json'
    $modelDir = Join-Path $AppDir 'runs\tuning\train'
    $target   = Join-Path $modelDir 'best_train_model.pt'
    $stamp    = Join-Path $modelDir 'MODEL_VERSION.txt'

    if (-not (Test-Path $manifest)) { return }   # nothing declares a model
    $m = Get-Content $manifest -Raw | ConvertFrom-Json
    $version = "$($m.version)".Trim()
    $url     = "$($m.url)".Trim()
    $sha     = ("$($m.sha256)".Trim().ToLower() -replace '^sha256:', '')
    if (-not $version -or -not $url) { return }   # no model published yet

    # Already up to date?
    if ((Test-Path $stamp) -and (Test-Path $target)) {
        if ((Get-Content $stamp -Raw).Trim() -eq $version) {
            Info "Counting model is up to date ($version)."
            return
        }
    }
    # Bundled model may already match (fresh install with no stamp) -- check hash
    # before downloading 160+ MB needlessly.
    if ($sha -and (Test-Path $target)) {
        $have = (Get-FileHash $target -Algorithm SHA256).Hash.ToLower()
        if ($have -eq $sha) {
            Set-Content -Path $stamp -Value $version -Encoding ascii
            Info "Counting model already current ($version)."
            return
        }
    }

    Info "Downloading the updated counting model ($version) - this is large, one time..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    $tmp = "$target.download"
    Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing

    if ($sha) {
        $got = (Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower()
        if ($got -ne $sha) {
            Remove-Item $tmp -Force -ErrorAction SilentlyContinue
            throw "checksum mismatch (expected $sha, got $got) - keeping the current model."
        }
    }

    Move-Item -Path $tmp -Destination $target -Force
    Set-Content -Path $stamp -Value $version -Encoding ascii
    Info "Counting model updated to $version."
}
catch {
    Write-Host ('  Model update skipped: ' + $_.Exception.Message)
    Write-Host '  The program still works with the model you have.'
}
