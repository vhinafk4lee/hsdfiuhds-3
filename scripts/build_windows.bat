@echo off
rem Build hcminer-gpu.exe on Windows.
rem Run from "x64 Native Tools Command Prompt for VS" so that cl.exe is on PATH.
setlocal
cd /d "%~dp0.."

where nvcc >nul 2>nul
if errorlevel 1 (
  echo [ERROR] nvcc not found.
  echo         Install the CUDA Toolkit - 12.8 or newer for RTX 50xx cards -
  echo         and reopen this window.
  exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
  echo [ERROR] cl.exe not found. nvcc needs the MSVC compiler.
  echo         Install "Desktop development with C++" from the Visual Studio Installer,
  echo         then use "x64 Native Tools Command Prompt for VS".
  exit /b 1
)

if "%CUDA_ARCH%"=="" (
  rem -arch=native builds for the card in this machine (CUDA 11.5+).
  set "ARCHFLAG=-arch=native"
) else (
  set "ARCHFLAG=-gencode arch=compute_%CUDA_ARCH%,code=sm_%CUDA_ARCH%"
)

echo Building with %ARCHFLAG% ...
nvcc -O3 -std=c++17 --use_fast_math -Xptxas -O3 %ARCHFLAG% ^
     -o src\cuda\hcminer-gpu.exe src\cuda\miner.cu
if errorlevel 1 (
  echo [ERROR] build failed.
  echo         If the card is newer than the toolkit, set CUDA_ARCH, e.g.:
  echo             set CUDA_ARCH=120 ^&^& scripts\build_windows.bat
  exit /b 1
)

echo.
echo Built src\cuda\hcminer-gpu.exe
src\cuda\hcminer-gpu.exe --list-devices
