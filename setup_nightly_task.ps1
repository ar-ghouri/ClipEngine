<#
.SYNOPSIS
    Registers the Windows Scheduled Task for ClipEngine Overnight Batch Clipping.
    
.DESCRIPTION
    Phase 1 Overnight Batch Pipeline Scheduler.
    Configures a daily 10:00 PM scheduled task named 'ClipEngineNightly' to execute
    nightly.py in standalone mode using the project's local virtual environment.
    
    SAFETY & ARCHITECTURAL CONFIGURATION:
    - WakeToRun: $true (Wakes the PC from sleep to execute batch runs)
    - StartWhenAvailable: $false (Strictly prevents missed-start catch-up runs during daytime)
    - ExecutionTimeLimit: 5 hours (Hard boundary exceeding nightly.py 4h/3:00 AM cutoff)
    - Working Directory: Explicitly set to project root
    
.NOTES
    DO NOT RUN THIS SCRIPT UNTIL YOU ARE READY FOR AUTOMATED OVERNIGHT RUNS.
    Run manually via PowerShell with Administrator privileges when desired:
    powershell -ExecutionPolicy Bypass -File .\setup_nightly_task.ps1
#>

[CmdletBinding()]
param(
    [string]$TaskName = "ClipEngineNightly",
    [string]$ProjectDir = "E:\NEW Projects\clipping",
    [string]$DailyTime = "10:00PM"
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " ClipEngine Scheduled Task Setup: $TaskName" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Validate Project Directory & Python Virtual Environment
if (-not (Test-Path $ProjectDir)) {
    Write-Error "Project directory does not exist: $ProjectDir"
}

$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$NightlyScript = "nightly.py"
$FullScriptPath = Join-Path $ProjectDir $NightlyScript

if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment Python executable not found: $PythonExe"
}

if (-not (Test-Path $FullScriptPath)) {
    Write-Error "Nightly pipeline script not found: $FullScriptPath"
}

Write-Host "✓ Project Root:      $ProjectDir" -ForegroundColor Green
Write-Host "✓ Python Executable: $PythonExe" -ForegroundColor Green
Write-Host "✓ Target Script:     $NightlyScript" -ForegroundColor Green
Write-Host "✓ Schedule:          Daily at $DailyTime" -ForegroundColor Green

# 2. Define Scheduled Task Action
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-u $NightlyScript" `
    -WorkingDirectory $ProjectDir

# 3. Define Daily Trigger at 10:00 PM
$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At $DailyTime

# 4. Define Task Settings (WakeToRun = True, StartWhenAvailable = False)
$Settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable:$false `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 5) `
    -RestartCount 0

# 5. Define Task Principal
$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType Interactive `
    -RunLevel Highest

# 6. Check for Existing Task & Register
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask) {
    Write-Host "! Existing task '$TaskName' found. Updating configuration..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "ClipEngine Overnight Batch & Source-Scouting Clipping Pipeline (10:00 PM Daily)"

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "✓ Task '$TaskName' registered successfully!" -ForegroundColor Green
Write-Host "  Schedule: Daily at $DailyTime" -ForegroundColor Green
Write-Host "  WakeToRun: Enabled | Missed-Start Catch-up: Disabled" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
