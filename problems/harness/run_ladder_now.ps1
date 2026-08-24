Set-Location (Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent)
try { & .\harness\monitor_sceladder.ps1 } catch { Write-Output ('CAUGHT: ' + $_.Exception.Message); Read-Host }
