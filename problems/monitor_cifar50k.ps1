# monitor_cifar50k.ps1 — long CIFAR run: wait for 50k CSV, then launch.
#   Goal: enough data + epochs for SGD to genuinely converge AND the
#   structural search (PATCH_POOLING) to have a chance to commit.
#   50k samples, 100 epochs, verbose, checkpointing (gpnn7).
$log = "monitor_50k_log.txt"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }

# ---- Stage 1: wait for prep (pid file) + CSV to exist ----
Log "stage1: waiting for 50k CSV prep"
$deadline = (Get-Date).AddMinutes(60)
while ((Get-Date) -lt $deadline) {
    if (Test-Path "bench_cifar_gray_50k.csv") { break }
    Start-Sleep -Seconds 30
}
if (-not (Test-Path "bench_cifar_gray_50k.csv")) {
    Log "stage1 FAILED: CSV never appeared"
    exit
}
$sz = [math]::Round((Get-Item "bench_cifar_gray_50k.csv").Length / 1MB, 1)
Log "stage1 done: CSV ready ($sz MB)"

# ---- Stage 2: launch the long run ----
Log "stage2: launching gray 50k, 100 epochs (gpnn7, checkpointing)"
$proc = Start-Process -FilePath "..\gpnn7.exe" -ArgumentList `
    '--csv','bench_cifar_gray_50k.csv','--input-cols','1024','--output-cols','10',`
    '--loss','bce','--max-epochs','100','--seed','1','--verbose',`
    '--save-interval','10' `
    -RedirectStandardOutput "bg_cifar50k_stdout.txt" `
    -RedirectStandardError "bg_cifar50k_stderr.txt" `
    -WindowStyle Hidden -PassThru
$proc.Id | Out-File "bg_cifar50k_pid.txt" -Encoding ascii
Log "stage2: pid=$($proc.Id) launched at $(Get-Date -Format 'HH:mm') — monitor exiting (run is self-contained + checkpoints)"
