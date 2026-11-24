# GeoSat - PowerShell Launcher
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "      GeoSat - Satellite Image Retrieval System         " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment (.venv) not found! Please run setup."
    exit 1
}

Write-Host "[1/2] Launching Backend REST API on http://127.0.0.1:5000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ScriptDir'; & '$PythonExe' backend\flask_backend.py"

Write-Host "[2/2] Launching Web UI Server on http://127.0.0.1:8080 (serving frontend/)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ScriptDir'; & '$PythonExe' -m http.server 8080 -d frontend"

Start-Sleep -Seconds 2
Write-Host "[OK] Opening GeoSat Web UI in browser..." -ForegroundColor Green
Start-Process "http://127.0.0.1:8080"

Write-Host "`nGeoSat is live!" -ForegroundColor Cyan
Write-Host "  * Backend API: http://127.0.0.1:5000/api/health" -ForegroundColor Gray
Write-Host "  * Frontend UI: http://127.0.0.1:8080" -ForegroundColor Gray
