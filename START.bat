@echo off
:: ============================================================
:: Options Signal Advisory System — Local Startup
:: Double-click this file to start the system.
:: ============================================================
title Options Signal Advisory System

cd /d "c:\Users\msi\OneDrive\Desktop\Share Trading"

echo.
echo  ============================================================
echo   OPTIONS SIGNAL ADVISORY SYSTEM  ^|  Personal Use Only
echo   NSE/BSE India  ^|  No Auto-Execution
echo  ============================================================
echo.

:: Activate virtual environment
call venv\Scripts\activate.bat

echo  [1] Choose mode:
echo      1 = Scanner Loop  (runs every 5 min during market hours)
echo      2 = Dashboard only (Streamlit at localhost:8501)
echo      3 = Both (scanner + dashboard in separate windows)
echo      4 = Dry Run (single pass, no Telegram send)
echo.
set /p MODE="  Enter choice [1-4]: "

if "%MODE%"=="1" goto SCANNER
if "%MODE%"=="2" goto DASHBOARD
if "%MODE%"=="3" goto BOTH
if "%MODE%"=="4" goto DRYRUN

:SCANNER
echo.
echo  Starting signal scanner loop...
echo  Press Ctrl+C to stop.
echo.
python main.py --loop
goto END

:DASHBOARD
echo.
echo  Starting Streamlit dashboard at http://localhost:8501
echo  Press Ctrl+C to stop.
echo.
start "" "http://localhost:8501"
python -m streamlit run dashboard/app.py --server.port 8501
goto END

:BOTH
echo.
echo  Starting scanner + dashboard in separate windows...
echo.
start "Signal Scanner" cmd /k "cd /d C:\Users\msi\OneDrive\Desktop\Share Trading && venv\Scripts\activate.bat && python main.py --loop"
timeout /t 3 >nul
start "Dashboard" cmd /k "cd /d C:\Users\msi\OneDrive\Desktop\Share Trading && venv\Scripts\activate.bat && python -m streamlit run dashboard/app.py --server.port 8501"
timeout /t 4 >nul
start "" "http://localhost:8501"
echo  Both windows started. Close them to stop.
goto END

:DRYRUN
echo.
echo  Running single dry-run pass (no Telegram send)...
echo.
python main.py --dry-run
echo.
echo  Dry run complete.
pause
goto END

:END
