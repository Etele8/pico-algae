; Inno Setup script for Pico-Algae Counter.
; Produces a normal Windows wizard installer (Setup.exe) that:
;   - installs per-user (no admin needed) into a writable folder,
;   - copies the app + the counting model,
;   - then runs setup_env.ps1 which auto-installs Python if missing and the
;     right CPU/GPU PyTorch build automatically,
;   - creates Start Menu + Desktop shortcuts and an uninstaller.
;
; Build with:  ISCC.exe installer\pico.iss   (run from the repo root)

#define AppName    "Pico-Algae Counter"
#define AppVersion "1.0.0"
#define Publisher  "Pico-Algae"
#define SrcRoot    ".."

[Setup]
AppId={{7B2F0B1A-9C4E-4E2E-9A9C-1D2E3F4A5B6C}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
; Per-user install into a writable location (the venv and captures need write access).
DefaultDirName={localappdata}\Programs\Pico-Algae Counter
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#SrcRoot}
OutputBaseFilename=Pico-Algae-Counter-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
LicenseFile={#SrcRoot}\LICENSE
ArchitecturesInstallIn64BitMode=x64compatible
DisableDirPage=no

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "{#SrcRoot}\Start Pico-Algae Counter.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcRoot}\Capture from Olympus.bat";    DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcRoot}\Install.bat";                 DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcRoot}\Update.bat";                  DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcRoot}\requirements.txt";            DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcRoot}\app\*";                       DestDir: "{app}\app";   Flags: ignoreversion recursesubdirs; Excludes: "__pycache__\*,*.pyc"
Source: "{#SrcRoot}\tools\*";                     DestDir: "{app}\tools"; Flags: ignoreversion recursesubdirs; Excludes: "__pycache__\*,*.pyc"
Source: "{#SrcRoot}\runs\tuning\train\best_train_model.pt"; DestDir: "{app}\runs\tuning\train"; Flags: ignoreversion

[Icons]
Name: "{group}\Pico-Algae Counter";      Filename: "{app}\Start Pico-Algae Counter.bat"; WorkingDir: "{app}"; Comment: "Count pico-algae in microscopy images"
Name: "{group}\Capture from Olympus";    Filename: "{app}\Capture from Olympus.bat";     WorkingDir: "{app}"; Comment: "Screenshot the microscope view and count"
Name: "{group}\Check for updates";       Filename: "{app}\Update.bat";                   WorkingDir: "{app}"; Comment: "Download the latest version"
Name: "{group}\Repair setup";            Filename: "{app}\Install.bat";                  WorkingDir: "{app}"; Comment: "Reinstall Python components"
Name: "{group}\Uninstall Pico-Algae Counter"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Pico-Algae Counter"; Filename: "{app}\Start Pico-Algae Counter.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: files;          Name: "{app}\app\VERSION.txt"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  RC: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption :=
      'Setting up Python and PyTorch. The first time this can take several minutes...';
    if not Exec('powershell.exe',
        '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\tools\setup_env.ps1') +
        '" -AppDir "' + ExpandConstant('{app}') + '"',
        '', SW_SHOW, ewWaitUntilTerminated, RC) then
      RC := -1;
    if RC <> 0 then
      MsgBox('The program files were installed, but setting up the Python components did not '
        + 'finish (code ' + IntToStr(RC) + ').' + #13#10 + #13#10
        + 'Make sure this PC has an internet connection, then use '
        + '"Repair setup" in the Start Menu (Pico-Algae Counter) to try again.',
        mbError, MB_OK);
  end;
end;
