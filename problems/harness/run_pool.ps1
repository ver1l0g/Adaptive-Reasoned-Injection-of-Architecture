# run_pool.ps1 — parallel job runner with a slot cap.
# Usage:  powershell -File harness\run_pool.ps1 -TaskFile harness\tasks.txt -Slots 5
# Task file: one shell command per line (# = comment). Each task runs via
# cmd /c from problems/ with its own redirection. The pool keeps at most
# -Slots tasks alive at once (each aria process measured <20% CPU, so 5
# slots saturate the machine without contention timeouts — the M6.7
# lesson: heavy 5-way overlap starves shadow validation watchdogs, so
# keep heavy runs at 3-4 and use 5 only for light suites).
param(
    [Parameter(Mandatory = $true)][string]$TaskFile,
    [int]$Slots = 5
)
Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent | Split-Path)
$tasks = Get-Content $TaskFile | Where-Object { $_ -and $_.Trim() -and ($_ -notmatch '^\s*#') }
Write-Host "run_pool: $($tasks.Count) tasks, $Slots slots"
$running = @()
$done = 0
foreach ($t in $tasks) {
    while ($running.Count -ge $Slots) {
        Start-Sleep -Seconds 10
        $running = @($running | Where-Object { -not $_.HasExited })
    }
    Write-Host "[$(Get-Date -Format HH:mm:ss)] launch: $t"
    $running += (Start-Process cmd -ArgumentList '/c', $t -WindowStyle Hidden -PassThru)
    Start-Sleep -Seconds 2
}
while ($running.Count -gt 0) {
    Start-Sleep -Seconds 10
    $running = @($running | Where-Object { -not $_.HasExited })
}
Write-Host "run_pool: all $($tasks.Count) tasks complete"
