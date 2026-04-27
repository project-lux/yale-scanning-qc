@echo off
REM ============================================================
REM  Yale Scanning QC - startup launcher (Windows)
REM
REM  Detects this machine's LAN IPv4, writes the access URL to
REM  logs\server-url.txt, then starts the QC server with all
REM  output (stdout + stderr) redirected to a timestamped log.
REM
REM  Run at boot via Task Scheduler (see README for steps), or
REM  double-click to launch manually.
REM ============================================================

setlocal EnableDelayedExpansion

REM --- Move to the directory containing this script ---
cd /d "%~dp0"

REM --- Ensure logs directory exists ---
if not exist "logs" mkdir "logs"

REM --- Build a YYYYMMDD_HHMMSS timestamp ---
for /f "skip=1 tokens=1" %%T in ('wmic os get localdatetime') do (
    if not defined RAW_TS set "RAW_TS=%%T"
)
set "TS=!RAW_TS:~0,8!_!RAW_TS:~8,6!"

set "LOGFILE=logs\qc-!TS!.log"
set "URLFILE=logs\server-url.txt"

REM --- Detect LAN IPv4 (first non-loopback, non-APIPA IPv4) ---
set "LAN_IP="
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /R /C:"IPv4 Address"') do (
    set "IP=%%A"
    set "IP=!IP: =!"
    if not "!IP!"=="127.0.0.1" (
        if not "!IP:~0,7!"=="169.254" (
            if not defined LAN_IP set "LAN_IP=!IP!"
        )
    )
)

if not defined LAN_IP set "LAN_IP=127.0.0.1"

set "PORT=5000"

REM --- Write the access URL to a fixed file + the log header ---
> "!URLFILE!" echo Yale Scanning QC
>>"!URLFILE!" echo --------------------------------------
>>"!URLFILE!" echo Started:    !TS!
>>"!URLFILE!" echo Access URL: http://!LAN_IP!:!PORT!
>>"!URLFILE!" echo Log file:   !LOGFILE!

> "!LOGFILE!" echo ============================================
>>"!LOGFILE!" echo Yale Scanning QC starting at !TS!
>>"!LOGFILE!" echo Access URL: http://!LAN_IP!:!PORT!
>>"!LOGFILE!" echo ============================================
>>"!LOGFILE!" echo.

echo.
echo Yale Scanning QC
echo --------------------------------------
echo Access URL: http://!LAN_IP!:!PORT!
echo Log file:   !LOGFILE!
echo URL file:   !URLFILE!
echo.

REM --- Launch the server, all output -> log file ---
uv run yale-qc serve --host 0.0.0.0 --port !PORT! >> "!LOGFILE!" 2>&1

endlocal
