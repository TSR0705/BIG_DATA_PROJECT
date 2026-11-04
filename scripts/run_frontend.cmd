@echo off
REM Start the CreditLens Next.js frontend (http://localhost:3000). Run from any folder.
cd /d "%~dp0..\frontend"
if not exist node_modules\react-plotly.js call npm install --no-audit --no-fund
call npm run dev
