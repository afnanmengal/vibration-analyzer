@echo off
setlocal enabledelayedexpansion
title GPU-Accelerated Mechanical Vibration Diagnostic Workstation

cd /d "%~dp0"

echo ===============================================================================
echo   GPU-ACCELERATED MECHANICAL VIBRATION ANALYZER ^& DIAGNOSTIC WORKSTATION
echo   PySide6 + PyQtGraph OpenGL + PortAudio C-Callback + Numba JIT Acceleration
echo ===============================================================================
echo.

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
    echo [ERROR] Python was not found on your system or in PATH.
    echo Please install Python 3.10+ or Anaconda/Miniconda to proceed.
    echo.
    pause
    exit /b 1
)

echo Python Environment Detected:
echo   !PYTHON_EXE!
echo.
echo ===============================================================================
echo   SELECT WORKSTATION LAUNCH MODE:
echo ===============================================================================
echo   [1] Launch 60 FPS GPU-Accelerated PySide6 Workstation [DEFAULT]
echo       * Multi-Pane QDockWidgets (Dockable, Floating, Tabbed)
echo       * Dual-Trace Oscilloscope with Rising-Edge Trigger ^& Phosphor Decay
echo       * Narrowband FFT with 1X-10X Harmonics, Peak-Hold, Sidebands ^& Ghost Baseline
echo       * Rolling 2D Spectrogram ^& 3D OpenGL Spectral Cascade Mesh
echo       * Hilbert Envelope Demodulation (HFRT) ^& Power Cepstrum
echo       * Flight Session Recorder (WAV + Synchronized CSV Logs)
echo.
echo   [2] Launch Reactive Web Dashboard (Streamlit + Plotly)
echo   [3] Run Automated DSP, Numba JIT ^& ML Anomaly Test Suite (pytest)
echo   [4] Install / Verify Workstation Dependencies (pip install requirements.txt)
echo   [5] Exit
echo ===============================================================================
echo.

choice /C 12345 /T 6 /D 1 /M "Select an option [1-5] (Auto-starting Option 1 in 6s): "
set CHOICE_VAL=!ERRORLEVEL!

echo.
if !CHOICE_VAL! equ 1 goto launch_workstation
if !CHOICE_VAL! equ 2 goto launch_web
if !CHOICE_VAL! equ 3 goto run_tests
if !CHOICE_VAL! equ 4 goto install_deps
if !CHOICE_VAL! equ 5 goto exit_app

:launch_workstation
echo -------------------------------------------------------------------------------
echo Launching GPU-Accelerated PySide6 Real-Time Diagnostic Workstation...
echo -------------------------------------------------------------------------------
"!PYTHON_EXE!" "%~dp0run_vibration_analyzer.py"
if !ERRORLEVEL! neq 0 (
    echo.
    echo [WARNING] Workstation terminated with error code !ERRORLEVEL!.
    pause
)
goto exit_app

:launch_web
echo -------------------------------------------------------------------------------
echo Launching Streamlit Web Dashboard...
echo -------------------------------------------------------------------------------
"!PYTHON_EXE!" "%~dp0run_vibration_analyzer.py" --web
if !ERRORLEVEL! neq 0 (
    echo.
    echo [WARNING] Web dashboard terminated with error code !ERRORLEVEL!.
    pause
)
goto exit_app

:run_tests
echo -------------------------------------------------------------------------------
echo Running Verification Test Suite (DSP, Numba JIT, Ring Buffers, ML Anomaly)...
echo -------------------------------------------------------------------------------
"!PYTHON_EXE!" -m pytest "%~dp0vibration_analyzer\tests" -v
echo.
pause
goto exit_app

:install_deps
echo -------------------------------------------------------------------------------
echo Installing and verifying workstation requirements...
echo -------------------------------------------------------------------------------
"!PYTHON_EXE!" -m pip install -r "%~dp0vibration_analyzer\requirements.txt"
echo.
echo Dependencies verified successfully.
pause
goto exit_app

:exit_app
exit /b 0
