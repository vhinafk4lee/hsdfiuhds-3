@echo off
rem Full Windows setup: python deps, GPU build, self-tests.
setlocal
cd /d "%~dp0.."

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python not found. Install Python 3.11+ and tick "Add to PATH".
  exit /b 1
)

echo == python dependencies ==
python -m pip install --quiet -r requirements.txt || exit /b 1

echo == building the CUDA miner ==
call scripts\build_windows.bat || exit /b 1

echo == self-tests ==
python tests\test_schema.py || exit /b 1
python tests\test_lane_splice.py || exit /b 1

echo.
echo Ready. Next:
echo   python -m hcminer.cli tune            ^(find the fastest settings for this card^)
echo   python -m hcminer.cli bench --seconds 20
