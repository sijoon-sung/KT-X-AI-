param(
  [ValidateSet('install','start','stop','status')][string]$Action='status',
  [string]$Config,
  [string]$Pythonw,
  [int]$Port=8772
)
$ErrorActionPreference='Stop'
$task='TashuExternalAssistantH27'
if($Action -eq 'install') {
  if(!(Test-Path -LiteralPath $Config) -or !(Test-Path -LiteralPath $Pythonw)){throw 'Private config and pythonw.exe are required.'}
  if($Port -lt 1024 -or $Port -gt 65535){throw 'Invalid loopback port.'}
  $script=Join-Path $PSScriptRoot 'external_proxy.py'
  $user=[Security.Principal.WindowsIdentity]::GetCurrent().Name
  $run=New-ScheduledTaskAction -Execute (Resolve-Path -LiteralPath $Pythonw).Path -Argument ('"'+$script+'" --config "'+(Resolve-Path -LiteralPath $Config).Path+'" --port '+$Port) -WorkingDirectory $PSScriptRoot
  $trigger=New-ScheduledTaskTrigger -AtLogOn -User $user
  $principal=New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
  $settings=New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
  Register-ScheduledTask -TaskName $task -Action $run -Trigger $trigger -Principal $principal -Settings $settings -Description 'Loopback Tashu display and official hosted LLM relay.' -Force | Out-Null
  Start-ScheduledTask -TaskName $task
} elseif($Action -eq 'start') {Start-ScheduledTask -TaskName $task}
elseif($Action -eq 'stop') {Stop-ScheduledTask -TaskName $task}
Get-ScheduledTask -TaskName $task | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName $task | Select-Object LastRunTime,LastTaskResult
