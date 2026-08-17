# watch_and_launch.ps1 — wait for baseline gpnn (pid file) to exit,
# then launch gpnn2 with patch pooling on the same dataset/seed.
$pidFile = "bg_cifar_pid.txt"
$baselinePid = Get-Content $pidFile

# Wait for baseline to finish (poll every 60s, max 4 hours)
$deadline = (Get-Date).AddHours(4)
while ((Get-Date) -lt $deadline) {
    $alive = Get-Process -Id $baselinePid -ErrorAction SilentlyContinue
    if (-not $alive) { break }
    Start-Sleep -Seconds 60
}

# Launch patch-pooling run: same dataset, same seed, more epochs so the
# plateau + structural search has room to trigger.
$proc = Start-Process -FilePath "..\gpnn2.exe" -ArgumentList `
    '--csv','bench_cifar_gray_5k.csv','--input-cols','1024','--output-cols','10',`
    '--loss','bce','--max-epochs','40','--seed','1','--verbose' `
    -RedirectStandardOutput "bg_cifar_pp_stdout.txt" `
    -RedirectStandardError "bg_cifar_pp_stderr.txt" `
    -WindowStyle Hidden -PassThru
$proc.Id | Out-File "bg_cifar_pp_pid.txt" -Encoding ascii
"launched pid=$($proc.Id) at $(Get-Date -Format 'HH:mm:ss')" | Out-File "watcher_done.txt" -Encoding ascii
