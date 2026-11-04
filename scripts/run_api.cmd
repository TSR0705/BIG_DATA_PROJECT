@echo off
REM Start the CreditLens FastAPI backend (http://localhost:8000). Run from any folder.
cd /d "%~dp0.."
call venv\Scripts\activate.bat
python -m uvicorn api.app.main:app --host 127.0.0.1 --port 8000 --reload
