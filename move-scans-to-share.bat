@echo off
REM ============================================================
REM  DSCA Scan Mover
REM
REM  Moves new scan files from the scanner workstation's Desktop
REM  to the DSCA AI Project share. Replaces the manual end-of-shift
REM  move that Ja'nya was doing by hand.
REM
REM  Usage: double-click this file (or a shortcut to it on the
REM  Desktop). A console window opens, shows a summary of what
REM  moved, then waits for the operator to press a key.
REM
REM  Safety:
REM    - ONLY image files (extensions in PATTERNS below) are touched.
REM    - Source files are deleted ONLY after a verified copy.
REM    - If the destination share is unreachable, the script
REM      aborts BEFORE moving anything.
REM ============================================================

setlocal EnableDelayedExpansion

REM ===================== CONFIG ===============================
REM Source: where Ja'nya saves new scans on DELL-i2S.
REM   Default: the whole Desktop (safe because PATTERNS filters
REM   to image types only). If she keeps scans in a subfolder,
REM   change this to e.g. "%USERPROFILE%\Desktop\Scans".
set "SOURCE=%USERPROFILE%\Desktop"

REM Destination: DSCA AI Project share.
REM   Use either a mapped drive letter ("Y:\AI Project") or the
REM   full UNC path ("\\fileserver.yale.edu\DSCA_Digital_Studio-CC1702-LIBITS\AI Project").
set "DEST=Y:\AI Project"

REM Only files matching these patterns will be moved.
set "PATTERNS=*.tif *.tiff *.jpg *.jpeg *.png *.cr2 *.cr3 *.nef *.dng *.arw *.raf"
REM ============================================================

cd /d "%~dp0"
if not exist "logs" mkdir "logs"

for /f "skip=1 tokens=1" %%T in ('wmic os get localdatetime') do (
    if not defined RAW_TS set "RAW_TS=%%T"
)
set "TS=!RAW_TS:~0,8!_!RAW_TS:~8,6!"
set "LOGFILE=logs\move-!TS!.log"

echo.
echo ============================================
echo   DSCA Scan Mover
echo ============================================
echo   From: !SOURCE!
echo   To:   !DEST!
echo   Log:  !LOGFILE!
echo ============================================
echo.

if not exist "!SOURCE!" (
    echo [ERROR] Source folder does not exist:
    echo         !SOURCE!
    echo.
    echo Edit the SOURCE line at the top of this script and try again.
    echo No files have been moved.
    echo.
    pause
    exit /b 1
)

if not exist "!DEST!" (
    echo [ERROR] Destination is not reachable:
    echo         !DEST!
    echo.
    echo Possible causes:
    echo   - The Y: drive is not connected. Open File Explorer and
    echo     reconnect it, or check VPN / network connection.
    echo   - The share path has changed. Edit the DEST line at the
    echo     top of this script.
    echo.
    echo No files have been moved.
    echo.
    pause
    exit /b 1
)

echo Moving image files...
echo (The first time you run this it may take a while. The window
echo  will pause when it's done so you can read the summary.)
echo.

REM robocopy flags:
REM   /MOV    move files (delete from source after verified copy)
REM   /Z      restartable mode (survives brief network drops)
REM   /R:3    retry up to 3 times on failure
REM   /W:5    wait 5 seconds between retries
REM   /NP     don't print per-file progress percentages
REM   /NDL    don't list every directory name
REM   /TEE    write to console AND log file
REM   /LOG+:  append to the log file
robocopy "!SOURCE!" "!DEST!" !PATTERNS! /MOV /Z /R:3 /W:5 /NP /NDL /TEE /LOG+:"!LOGFILE!"
set "RC=!ERRORLEVEL!"

echo.
echo ============================================
if !RC! LEQ 1 (
    echo   DONE. All matching files were moved.
) else if !RC! LEQ 7 (
    echo   DONE with warnings. Some files were skipped --
    echo   open the log to see which ones.
) else (
    echo   [ERROR] Robocopy reported a serious problem.
    echo   Exit code: !RC!
    echo   Open the log file for details. Some files may not
    echo   have been moved -- DO NOT delete the originals.
)
echo   Log: !LOGFILE!
echo ============================================
echo.
pause

endlocal
