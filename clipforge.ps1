<#
.SYNOPSIS
    Run ClipForge without activating the virtual environment.

.DESCRIPTION
    Finds the venv relative to this script, so it works from any directory and
    does not care what is on PATH. With no arguments it opens the app, which is
    what `ClipForge.cmd` does on a double-click.

    Deliberately a script rather than a frozen executable - see ClipForge.cmd
    for the reasoning.

    ASCII only, and CRLF line endings, for the reasons recorded in
    ClipForge.cmd and .gitattributes. Windows PowerShell 5.1 reads an unsigned
    script in the ANSI codepage, not UTF-8.

.EXAMPLE
    .\clipforge.ps1
    Opens the app in a browser.

.EXAMPLE
    .\clipforge.ps1 export 2026-08-14_marvel --min-rating 1
    Writes an FCPXML timeline of everything rated 'maybe' or better.

.EXAMPLE
    .\clipforge.ps1 doctor
    Checks ffmpeg, the Python environment, and the chosen proxy encoder.
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$clipforge = Join-Path $root '.venv\Scripts\clipforge.exe'

if (-not (Test-Path $clipforge)) {
    Write-Host ''
    Write-Host '  ClipForge is not set up in this folder yet.' -ForegroundColor Yellow
    Write-Host ''
    Write-Host "  Expected: $clipforge"
    Write-Host ''
    Write-Host '  Run:'
    Write-Host '      py -3.12 -m venv .venv'
    Write-Host '      .venv\Scripts\python -m pip install -e ".[dev]"'
    Write-Host '      .venv\Scripts\clipforge doctor'
    Write-Host ''
    Write-Host '  See README.md for the full setup.'
    Write-Host ''
    exit 1
}

# No arguments is the double-click case: open the app.
if (-not $Arguments -or $Arguments.Count -eq 0) {
    $Arguments = @('review')
}

# Run from the project root. `config.find_project_root` walks up from the
# working directory to locate pyproject.toml, and relative paths like
# `paths.data_root: ./data` resolve against what it finds, so invoking this
# from elsewhere would otherwise warn and address a different database.
Push-Location $root
try {
    & $clipforge @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
