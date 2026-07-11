# Desktop Executable (Frontend + Backend)

This project can be packaged as a single Windows executable that includes:

- Flask backend
- Built Vue frontend
- Local reference data files used by OCR and parsing

## Build

Run from my-grocery-backend:

```powershell
./build_desktop.ps1
```

Optional: provide a different API base URL (for example Cloud Run) for desktop build:

```powershell
./build_desktop.ps1 -ApiBaseUrl "https://YOUR-SERVICE-URL/api"
```

## Output

The executable is created at:

- `my-grocery-backend/dist/MyGroceryHome.exe`

## Build Installer (Setup.exe)

You can generate an installer package using Inno Setup.

Prerequisite:

- Install Inno Setup 6 (compiler `ISCC.exe`).

Run from my-grocery-backend:

```powershell
./build_installer.ps1
```

Optional: skip rebuilding the desktop executable if it already exists:

```powershell
./build_installer.ps1 -SkipDesktopBuild
```

Optional: set a custom path to `ISCC.exe`:

```powershell
./build_installer.ps1 -InnoSetupCompilerPath "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
```

Installer output:

- `my-grocery-backend/installer-output/MyGroceryHomeSetup.exe`

Installer behavior for OCR prerequisite:

- Detects whether Tesseract OCR is installed.
- If missing, shows an informational message during install.
- Adds an optional final-page checkbox to open the official Tesseract download page.

## Run on Other Devices

1. Copy `MyGroceryHome.exe` to the target Windows device.
2. Run the executable.
3. If OCR features do not work, install Tesseract OCR on that device and add it to PATH.

## Notes

- Desktop mode opens the local bundled server URL and serves the bundled frontend.
- By default, API calls from the bundled frontend use the local bundled backend (`/api`).
