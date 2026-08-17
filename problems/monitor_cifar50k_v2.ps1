# monitor_cifar50k_v2.ps1 — relaunch after prep FULLY completes.
# Completion = python prep process gone AND "Wrote" line in prep stdout
# AND CSV size stable for 2 consecutive checks.
$log = "monitor_50k_log.txt"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }

Log "v2 armed: waiting for prep completion"

# Wait for prep python to exit (pid file), max 60 min
$prepPid = Get-Content "bg_prep50k_pid.txt" -ErrorAction SilentlyContinue
$deadline = (Get-Date).AddMinutes(60)
while ((Get-Date) -lt $deadline) {
    if ($prepPid -and -not (Get-Process -Id $prepPid -ErrorAction SilentlyContinue)) { break }
    if (-not $prepPid) { break }
    Start-Sleep -Seconds 30
}
Log "prep process exited"

# Verify completion marker + size stability
$ok = $false
$deadline = (Get-Date).AddMinutes(10)
$lastSize = -1
while ((Get-Date) -lt $deadline) {
    $wrote = Select-String -Path "bg_prep50k_stdout.txt" -Pattern "Wrote" -ErrorAction SilentlyContinue
    $sz = (Get-Item "bench_cifar_gray_50k.csv" -ErrorAction SilentlyContinue).Length
    if ($wrote -and $sz -eq $lastSize -and $sz -gt 0) { $ok = $true; break }
    $lastSize = $sz
    Start-Sleep -Seconds 20
}
if (-not $ok) {
    Log "v2 FAILED: prep did not complete cleanly"
    exit
}
Log "prep verified complete ($([math]::Round($lastSize/1MB,1)) MB stable)"

# Launch the long run
$proc = Start-Process -FilePath "..\gpnn7.exe" -ArgumentList `
    '--csv','bench_cifar_gray_50k.csv','--input-cols','1024','--output-cols','10',`
    '--loss','bce','--max-epochs','100','--seed','1','--verbose',`
    '--save-interval','10' `
    -RedirectStandardOutput "bg_cifar50k_stdout.txt" `
    -RedirectStandardError "bg_cifar50k_stderr.txt" `
    -WindowStyle Hidden -PassThru
$proc.Id | Out-File "bg_cifar50k_pid.txt" -Encoding ascii
Log "v2 launched pid=$($proc.Id) at $(Get-Date -Format 'HH:mm')"
