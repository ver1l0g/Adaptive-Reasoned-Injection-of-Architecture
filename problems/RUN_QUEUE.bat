@echo off
rem RUN_QUEUE.bat — double-click to start the ARIA overnight queue.
rem Keep the console window OPEN (minimize is fine). Close it = stop the queue
rem (completed stages are remembered; re-running resumes where it left off).
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File runner.ps1
pause
