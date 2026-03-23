# Creates .venv in the project root. Run from project root:
#   powershell -ExecutionPolicy Bypass -File scripts\create_venv.ps1
$ErrorActionPreference = "Stop"
# Project root = folder that contains scripts/ (parent of this script's directory)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $ProjectRoot "requirements.txt"))) {
    $ProjectRoot = Get-Location
}

function Find-Python {
    foreach ($cmd in @("py", "python", "python3")) {
        try {
            $null = Get-Command $cmd -ErrorAction Stop
            return $cmd
        } catch { }
    }
    return $null
}

$pythonCmd = Find-Python
if (-not $pythonCmd) {
    Write-Host "ERROR: No Python found. Install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "During setup, check 'Add python.exe to PATH'. Then reopen PowerShell and run this script again." -ForegroundColor Yellow
    exit 1
}

$venvPath = Join-Path $ProjectRoot ".venv"
Write-Host "Using: $pythonCmd"
Write-Host "Creating venv at: $venvPath"

if ($pythonCmd -eq "py") {
    & py -3 -m venv $venvPath
} else {
    & $pythonCmd -m venv $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Host "ERROR: venv was not created." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done. Activate with:" -ForegroundColor Green
Write-Host "  cd `"$ProjectRoot`""
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then: pip install -r requirements.txt"
