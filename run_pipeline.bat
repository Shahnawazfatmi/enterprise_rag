@echo off

cd /d C:\Users\Lenovo\Desktop\consiva-ai-assistant

call venv\Scripts\activate.bat

python ingestion\pipeline.py

exit /b %ERRORLEVEL%