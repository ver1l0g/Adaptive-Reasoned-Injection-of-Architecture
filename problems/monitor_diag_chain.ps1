# monitor_diag_chain.ps1 — architectural-limit decision tests.
#   Waits for the rerun chain (Feynman/Korns/multiseed/charLM) to finish,
#   then runs with gpnn8 (softmax-CE eval):
#     D1: shakespeare w1 (bigram task, floor 3.538 bits/char), 30ep
#     D2: CIFAR gray 5k pre-pooled to 8x8 (64 inputs), 40ep — compare vs
#         raw 1024-input baseline (1.919 train / 3.42 val)
$log = "DIAG_CHAIN_LOG.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }

function Wait-ByPidFile($pidFile, $maxMinutes) {
    $deadline = (Get-Date).AddMinutes($maxMinutes)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path $pidFile) {
            $p = Get-Content $pidFile -ErrorAction SilentlyContinue
            if ($p -and -not (Get-Process -Id $p -ErrorAction SilentlyContinue)) { return $true }
        }
        Start-Sleep -Seconds 60
    }
    return $false
}

Log "diag chain armed — waiting for rerun chain"
[void](Wait-ByPidFile "bg_shakespeare_r_pid.txt" 1200)
Log "rerun chain finished; launching diagnostics"

# ---- D1: bigram ----
Log "D1: shakespeare w1 (bigram floor 3.538 bits/char)"
$p = Start-Process -FilePath "..\gpnn8.exe" -ArgumentList `
    '--csv','bench_shakespeare_w1.csv','--input-cols','1','--output-cols','65',`
    '--loss','bce','--max-epochs','30','--seed','1','--save-graph','none',`
    '--eval-csv','bench_shakespeare_w1.csv' `
    -RedirectStandardOutput "bg_diag_w1_stdout.txt" `
    -RedirectStandardError "bg_diag_w1_stderr.txt" `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File "bg_diag_w1_pid.txt" -Encoding ascii
[void](Wait-ByPidFile "bg_diag_w1_pid.txt" 360)
$sm = Select-String -Path "bg_diag_w1_stdout.txt" -Pattern "SoftmaxCE|Accuracy" |
      Select-Object -Last 2
Log "D1 done:`n$($sm -join "`n")"

# ---- D2: pre-pooled CIFAR ----
Log "D2: CIFAR gray 5k pre-pooled 8x8 (64 inputs)"
$p = Start-Process -FilePath "..\gpnn8.exe" -ArgumentList `
    '--csv','bench_cifar_gray_5k_pooled8.csv','--input-cols','64','--output-cols','10',`
    '--loss','bce','--max-epochs','40','--seed','1','--verbose' `
    -RedirectStandardOutput "bg_diag_pooled_stdout.txt" `
    -RedirectStandardError "bg_diag_pooled_stderr.txt" `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File "bg_diag_pooled_pid.txt" -Encoding ascii
[void](Wait-ByPidFile "bg_diag_pooled_pid.txt" 480)
$vl = Select-String -Path "bg_diag_pooled_stdout.txt" -Pattern "Restored best-val" |
      Select-Object -Last 1
Log "D2 done: $vl"
Log "=== DIAG CHAIN COMPLETE ==="
