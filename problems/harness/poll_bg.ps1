# poll_bg.ps1 — check status of a background run.
# Usage: .\poll_bg.ps1 <tag>
param([Parameter(Mandatory=$true)][string]$Tag)
$pidFile = "bg_${Tag}_pid.txt"
if (-not (Test-Path $pidFile)) { Write-Output "no such tag: $Tag"; exit }
$p = Get-Content $pidFile
$alive = Get-Process -Id $p -ErrorAction SilentlyContinue
if ($alive) {
    Write-Output "RUNNING pid=$p cpu=$([math]::Round($alive.CPU,1))s mem=$([math]::Round($alive.WorkingSet64/1MB,0))MB"
} else {
    Write-Output "FINISHED pid=$p"
}
$stdout = "bg_${Tag}_stdout.txt"
if (Test-Path $stdout) {
    Write-Output "--- last 5 stdout lines ---"
    Get-Content $stdout -Tail 5 -ErrorAction SilentlyContinue
}
