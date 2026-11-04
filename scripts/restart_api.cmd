@echo off
REM Stop whatever is listening on port 8000, then start the CreditLens API from this project.
cd /d "%~dp0.."
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /PID %%p /F >nul 2>&1
call venv\Scripts\activate.bat
echo Starting CreditLens API on http://localhost:8000 ...
python -m uvicorn api.app.main:app --host 127.0.0.1 --port 8000 --reload
