param(
    [string]$ApiBaseUrl = "/api"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$backendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendRoot = Join-Path $backendRoot "..\my-grocery-frontend"
$venvPython = Join-Path $backendRoot ".venv\Scripts\python.exe"
$frontendDist = Join-Path $frontendRoot "dist"
$backendDist = Join-Path $backendRoot "dist"

if (-not (Test-Path $frontendRoot)) {
    throw "Frontend folder not found at $frontendRoot"
}

if (-not (Test-Path $venvPython)) {
    throw "Python virtual environment not found. Create it first at my-grocery-backend/.venv"
}

Write-Host "Building frontend for desktop API target..."
Push-Location $frontendRoot
try {
    npm install
    $env:VUE_APP_API_BASE_URL = $ApiBaseUrl
    npm run build
}
finally {
    Pop-Location
}

if (-not (Test-Path $frontendDist)) {
    throw "Frontend build output not found at $frontendDist"
}

Write-Host "Syncing frontend build into backend..."
if (Test-Path $backendDist) {
    Remove-Item -Path $backendDist -Recurse -Force
}
New-Item -ItemType Directory -Path $backendDist | Out-Null
Copy-Item -Path (Join-Path $frontendDist "*") -Destination $backendDist -Recurse -Force

Write-Host "Installing PyInstaller in backend venv (if needed)..."
& $venvPython -m pip install pyinstaller

Write-Host "Creating one-file executable..."
Push-Location $backendRoot
try {
    & $venvPython -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --name MyGroceryHome `
        --add-data "dist;dist" `
        --add-data "Data-Folder;Data-Folder" `
        --add-data "Irrelevant.txt;." `
        --add-data "ItemCost.txt;." `
        --add-data "items_expiry.txt;." `
        --add-data "Kitchen_Eatables_Database.txt;." `
        --add-data "NonFoodItems.txt;." `
        --collect-all webview `
        app.py
}
finally {
    Pop-Location
}

Write-Host "Done. Executable generated at my-grocery-backend/dist/MyGroceryHome.exe"
Write-Host "If OCR fails on another device, install Tesseract OCR and ensure it is on PATH."