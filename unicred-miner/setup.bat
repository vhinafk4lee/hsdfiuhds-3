@echo off
setlocal
cd /d "%~dp0"
title UNICRED miner - setup
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto nopython

echo.
echo === 1/4 Installing Python packages ===
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto pipfail

echo.
echo === 2/4 config.json ===
if not exist config.json copy config.example.json config.json >nul
echo OK

echo.
echo === 3/4 servers.txt ===
if not exist servers.txt if exist "%USERPROFILE%\servers.txt" move "%USERPROFILE%\servers.txt" servers.txt >nul
if exist servers.txt goto haveservers
> servers.txt echo # Paste the SSH lines from vast.ai below, one per line
echo Notepad opens servers.txt: paste the SSH lines from vast.ai, one per line,
echo press Ctrl+S, close Notepad and come back to this window.
notepad servers.txt
pause
:haveservers
echo OK

echo.
echo === 4/4 wallet.key ===
if not exist wallet.key if exist "%USERPROFILE%\wallet.key" move "%USERPROFILE%\wallet.key" wallet.key >nul
if exist wallet.key goto havekey
type nul > wallet.key
echo Notepad opens wallet.key: paste the PRIVATE KEY of the mining wallet as one line,
echo press Ctrl+S, close Notepad and come back to this window.
echo Never send this key to anyone.
notepad wallet.key
pause
:havekey
echo OK

echo.
echo === Online check ===
%PY% unicred.py check
echo.
echo Next step: double-click servers.bat
pause
exit /b 0

:pipfail
echo pip install failed, see the messages above.
echo If the Microsoft Store opened instead, Python is not installed: see python.org
pause
exit /b 1

:nopython
echo Python not found. Install Python 3.9+ from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer, then run setup.bat again.
pause
exit /b 1
