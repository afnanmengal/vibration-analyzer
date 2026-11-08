@echo off
title Mechanical Vibration Analyzer Web Dashboard
echo ===============================================================================
echo   Mechanical Vibration Analyzer - Streamlit Web Dashboard
echo ===============================================================================
echo.
echo Launching Streamlit Web Dashboard...
python "%~dp0run_vibration_analyzer.py" --web
pause
