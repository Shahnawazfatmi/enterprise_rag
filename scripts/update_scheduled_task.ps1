# Fix the nightly ingestion task so Windows stops killing it.
# Run once in an elevated PowerShell (Run as administrator):
#   powershell -ExecutionPolicy Bypass -File scripts\update_scheduled_task.ps1

$taskName = "Consiva AI - Nightly Ingestion"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -MultipleInstances IgnoreNew

Set-ScheduledTask -TaskName $taskName -Settings $settings | Out-Null

(Get-ScheduledTask -TaskName $taskName).Settings |
    Format-List DisallowStartIfOnBatteries, StopIfGoingOnBatteries, WakeToRun,
                StartWhenAvailable, ExecutionTimeLimit, MultipleInstances
