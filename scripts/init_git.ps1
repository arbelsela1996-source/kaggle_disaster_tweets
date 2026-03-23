# Run once if the repo needs a fresh Git setup (requires Git for Windows on PATH).
# https://git-scm.com/download/win
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Install Git from https://git-scm.com/download/win then run this again." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path .git)) { git init -b main }
git add -A
git status
Write-Host "`nTo create first commit: git commit -m `"Initial commit`"" -ForegroundColor Green
Write-Host "Then: git remote add origin <your-github-url> && git push -u origin main" -ForegroundColor Green
