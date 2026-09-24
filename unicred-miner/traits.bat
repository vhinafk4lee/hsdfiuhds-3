@echo off
setlocal
cd /d "%~dp0"
title UNICRED miner - traits
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto nopython
%PY% unicred.py traits
echo.
pause
exit /b 0

:nopython
echo Python not found. Install Python 3.9+ from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer, then run setup.bat again.
pause
exit /b 1
