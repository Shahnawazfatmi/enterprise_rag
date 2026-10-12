@echo off

cd /d "%~dp0"

rem UTF-8 everywhere: the Windows console code page (cp1252) crashes on
rem characters such as the rupee sign when a stage prints them.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

call venv\Scripts\activate.bat

python ingestion\pipeline.py

exit /b %ERRORLEVEL%
