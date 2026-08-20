# launch_bg.ps1 — start a gpnn run in the background, return immediately.
# Usage: .\launch_bg.ps1 <tag> <args...>
# Writes: bg_<tag>_stdout.txt, bg_<tag>_stderr.txt, bg_<tag>_pid.txt
param(
    [Parameter(Mandatory=$true)][string]$Tag,
    [Parameter(Mandatory=$true)][string[]]$GpnnArgs
)
$exe = "..\gpnn.exe"
$stdout = "bg_${Tag}_stdout.txt"
$stderr = "bg_${Tag}_stderr.txt"
# Clear old logs
Remove-Item $stdout, $stderr, "bg_${Tag}_pid.txt" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $exe -ArgumentList $GpnnArgs `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -WindowStyle Hidden -PassThru
$proc.Id | Out-File "bg_${Tag}_pid.txt" -Encoding ascii
Write-Output "started tag=$Tag pid=$($proc.Id)"
