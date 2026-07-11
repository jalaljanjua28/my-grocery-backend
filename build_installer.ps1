param(
    [string]$ApiBaseUrl = "/api",
    [switch]$SkipDesktopBuild,
    [string]$InnoSetupCompilerPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$backendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktopExePath = Join-Path $backendRoot "dist\MyGroceryHome.exe"
$issPath = Join-Path $backendRoot "installer\MyGroceryHome.iss"

if (-not (Test-Path $issPath)) {
    throw "Installer definition not found at $issPath"
}

if (-not $SkipDesktopBuild) {
    Write-Host "Building desktop executable first..."
    & (Join-Path $backendRoot "build_desktop.ps1") -ApiBaseUrl $ApiBaseUrl
}

if (-not (Test-Path $desktopExePath)) {
    throw "Desktop executable not found at $desktopExePath. Run ./build_desktop.ps1 first."
}

if (-not $InnoSetupCompilerPath) {
    $programFilesX86 = ${env:ProgramFiles(x86)}
    $programFiles = $env:ProgramFiles
    $localAppData = $env:LOCALAPPDATA

    $candidates = @(
        "$programFilesX86\Inno Setup 6\ISCC.exe",
        "$programFiles\Inno Setup 6\ISCC.exe",
        "$localAppData\Programs\Inno Setup 6\ISCC.exe"
    )

    $InnoSetupCompilerPath = ($candidates | Where-Object { Test-Path $_ } | Select-Object -First 1)
}

if (-not $InnoSetupCompilerPath -or -not (Test-Path $InnoSetupCompilerPath)) {
    throw "Inno Setup compiler not found. Install Inno Setup 6 and retry, or pass -InnoSetupCompilerPath."
}

Write-Host "Building installer with Inno Setup..."
Push-Location $backendRoot
try {
    & $InnoSetupCompilerPath $issPath
}
finally {
    Pop-Location
}

Write-Host "Done. Installer generated in my-grocery-backend/installer-output"