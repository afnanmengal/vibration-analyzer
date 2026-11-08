@echo off
setlocal enabledelayedexpansion
title Industrial Mechanical Vibration Analyzer

cd /d "%~dp0"

:: Detect Python Environment
set "PYTHON_EXE="
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
) else if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
) else (
    where python >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        for /f "delims=" %%i in ('where python') do (
            if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
        )
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python not found. Please install Python 3.10+ or Miniconda.
    pause
    exit /b 1
)

echo Starting 60 FPS Native Desktop Instrument...
"!PYTHON_EXE!" "%~dp0app.py"
if !ERRORLEVEL! neq 0 (
    echo.
    echo Application exited with error code !ERRORLEVEL!.
    pause
)
