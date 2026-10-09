@echo off
rem One-command launcher for Windows. Double-click this file, or run it from a terminal.
rem   run.bat           demo mode, no training needed
rem   run.bat trained   load the trained ensemble from artifacts\
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto :nopython

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 goto :nopython
)
call ".venv\Scripts\activate.bat"

if /i "%~1"=="trained" (
  echo Installing full requirements, this can take several minutes...
  python -m pip install -q -r requirements.txt
  set "DEMO="
) else (
  rem Demo mode needs only the light dependencies, not torch or the boosting libraries.
  python -m pip install -q flask flask-cors numpy
  set "DEMO=1"
)
if errorlevel 1 (
  echo.
  echo Package installation failed. Check your internet connection and try again.
  pause
  exit /b 1
)

if not defined PORT set "PORT=5050"
echo.
echo   URL Inspector running at  http://localhost:%PORT%
echo   Keep this window open. Press Ctrl+C to stop.
echo.
start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process 'http://localhost:%PORT%'"
python -m backend.app
if errorlevel 1 pause
exit /b

:nopython
echo.
echo Python 3 was not found.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH"
echo during setup, then run this file again.
echo.
pause
exit /b 1
