@echo off
title PDF Text, Font & Resolution Studio
echo Starting PDF Text, Font ^& Resolution Studio on http://127.0.0.1:8765 ...
start "" http://127.0.0.1:8765
python -m uvicorn app:app --app-dir "%~dp0pdf_studio" --host 127.0.0.1 --port 8765
pause
