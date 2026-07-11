#define MyAppName "My Grocery Home"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "My Grocery Home"
#define MyAppExeName "MyGroceryHome.exe"
#define TesseractDownloadUrl "https://github.com/UB-Mannheim/tesseract/wiki"

[Setup]
AppId={{4EBFB4A6-4074-4A8D-9BF2-4A20857BF92D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\My Grocery Home
DefaultGroupName=My Grocery Home
OutputDir=..\installer-output
OutputBaseFilename=MyGroceryHomeSetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\My Grocery Home"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\My Grocery Home"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch My Grocery Home"; Flags: nowait postinstall skipifsilent
Filename: "{#TesseractDownloadUrl}"; Description: "Open Tesseract OCR download page (recommended for receipt scanning)"; Flags: shellexec postinstall skipifsilent unchecked; Check: IsTesseractMissing

[Code]
function IsTesseractInstalled(): Boolean;
var
	PathValue: string;
	Candidate: string;
begin
	Result := False;

	Candidate := ExpandConstant('{pf}\Tesseract-OCR\tesseract.exe');
	if FileExists(Candidate) then
	begin
		Result := True;
		exit;
	end;

	Candidate := ExpandConstant('{pf32}\Tesseract-OCR\tesseract.exe');
	if FileExists(Candidate) then
	begin
		Result := True;
		exit;
	end;

	Candidate := ExpandConstant('{pf64}\Tesseract-OCR\tesseract.exe');
	if FileExists(Candidate) then
	begin
		Result := True;
		exit;
	end;

	if RegQueryStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', PathValue) then
	begin
		if Pos('Tesseract-OCR', PathValue) > 0 then
		begin
			Result := True;
			exit;
		end;
	end;

	if RegQueryStringValue(HKCU, 'Environment', 'Path', PathValue) then
	begin
		if Pos('Tesseract-OCR', PathValue) > 0 then
		begin
			Result := True;
			exit;
		end;
	end;
end;

function IsTesseractMissing(): Boolean;
begin
	Result := not IsTesseractInstalled();
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
	if (CurStep = ssInstall) and IsTesseractMissing() then
		MsgBox(
			'Tesseract OCR was not detected. The app will install and run, but receipt OCR may fail until Tesseract OCR is installed.' + #13#10 + #13#10 +
			'On the final page you can open the official download page.',
			mbInformation,
			MB_OK
		);
end;