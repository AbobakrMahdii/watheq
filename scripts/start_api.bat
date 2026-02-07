@echo off
setlocal
set ROOT=%~dp0..\
rem Ensure we run from project root (this .bat sits in scripts/)
cd /d %ROOT%
call "%ROOT%\.venv\Scripts\python.exe" -m api.main
endlocal
