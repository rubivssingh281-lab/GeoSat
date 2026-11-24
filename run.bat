@echo off
setlocal enabledelayedexpansion
title GeoSat Launcher
cd /d "%~dp0"

echo ========================================================
echo       GeoSat - Satellite Image Retrieval System
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment (.venv) not found!
    echo Please create it with uv or python.
    pause
    exit /b 1
)

echo [1/2] Starting GeoSat Backend API on http://127.0.0.1:5000 ...
start "GeoSat Backend API" cmd /k ".\.venv\Scripts\python.exe backend\flask_backend.py"

echo [2/2] Starting GeoSat Web UI on http://127.0.0.1:8080 (serving frontend/) ...
start "GeoSat Web UI" cmd /k ".\.venv\Scripts\python.exe -m http.server 8080 -d frontend"

echo.
echo Launching your browser at http://127.0.0.1:8080 ...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8080

echo.
echo [LIVE] GeoSat System is running!
echo   * Backend REST API:  http://127.0.0.1:5000
echo   * Web Frontend UI:   http://127.0.0.1:8080 (frontend/index.html)
echo.
echo Keep the server console windows open while using the application.
echo Press any key to exit this launcher window...
pause >nul
