@echo off
echo Stopping any existing Python processes...
taskkill /f /im python.exe 2>nul

echo Starting unified robot controller...
python unified_app.py

pause