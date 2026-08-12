@echo off
rem ClipForge - double-click this to open the app.
rem
rem Deliberately a script rather than a frozen executable. Once Phase 2 lands, a
rem PyInstaller build has to bundle torch: 3-5 GB, routinely quarantined by
rem antivirus, rebuilt on every dependency change, and it would STILL need
rem ffmpeg and the Whisper models fetched separately. It adds a build step
rem without removing a setup step.
rem
rem Any arguments are passed straight through, so this doubles as a CLI entry
rem point: ClipForge.cmd status, ClipForge.cmd export STREAM_ID, and so on.
rem
rem TWO RULES FOR EDITING THIS FILE, both learned the hard way:
rem
rem   1. CRLF line endings, always. cmd.exe misparses an LF-only batch file in a
rem      way that still runs -- it split a later "-m" into a phantom command and
rem      printed "'m' is not recognized" before carrying on regardless.
rem      .gitattributes pins this so a clone cannot undo it.
rem   2. ASCII only, including comments. Batch files are read in the console's
rem      OEM codepage, not UTF-8, so anything else arrives as mojibake -- and no
rem      angle brackets even in comments, because redirection is parsed before
rem      rem swallows the line.

setlocal
cd /d "%~dp0"

set "CLIPFORGE=%~dp0.venv\Scripts\clipforge.exe"

if not exist "%CLIPFORGE%" (
    echo.
    echo   ClipForge is not set up in this folder yet.
    echo.
    echo   Expected: %CLIPFORGE%
    echo.
    echo   From a PowerShell prompt here, run:
    echo.
    echo       py -3.12 -m venv .venv
    echo       .venv\Scripts\python -m pip install -e ".[dev]"
    echo       .venv\Scripts\clipforge doctor
    echo.
    echo   See README.md for the full setup.
    echo.
    pause
    exit /b 1
)

rem No arguments means the operator double-clicked it: open the app.
if "%~1"=="" (
    "%CLIPFORGE%" review
) else (
    "%CLIPFORGE%" %*
)

rem Hold the window open on failure so a double-click shows the reason instead
rem of flashing a console and vanishing.
if errorlevel 1 (
    echo.
    echo   ClipForge exited with an error. See above.
    pause
)

endlocal
