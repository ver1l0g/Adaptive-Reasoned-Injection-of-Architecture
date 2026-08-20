# monitor_gen_chain2.ps1 — continuation after reboot.
#   Stage 2 (already restarted externally): wait for gray 5k val-selection
#   Stage 3: gray 20k val-selection
#   Stage 4: report -> cifar_gen_report.txt
$log = "monitor_gen_log.txt"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }

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

function Get-FinalLoss($file) {
    if (-not (Test-Path $file)) { return "n/a" }
    $m = Select-String -Path $file -Pattern "Final loss:\s*([0-9.eE+-]+)" | Select-Object -Last 1
    if ($m) { return $m.Matches[0].Groups[1].Value } else { return "n/a" }
}

function Get-BestVal($file) {
    if (-not (Test-Path $file)) { return "n/a" }
    $m = Select-String -Path $file -Pattern "Restored best-val graph snapshot \(val=([0-9.eE+-]+)" | Select-Object -Last 1
    if ($m) { return $m.Matches[0].Groups[1].Value } else { return "n/a" }
}

Log "chain2 armed (post-reboot restart)"

# ---- Stage 2: gray 5k (restarted externally) ----
Log "stage2: waiting for restarted gray 5k val-selection (max 6h)"
[void](Wait-ByPidFile "bg_cifar_gray_vs_pid.txt" 360)
Log "stage2 done: final=$(Get-FinalLoss 'bg_cifar_gray_vs_stdout.txt') bestval=$(Get-BestVal 'bg_cifar_gray_vs_stdout.txt')"

# ---- Stage 3: gray 20k ----
if (Test-Path "bench_cifar_gray_20k.csv") {
    Log "stage3: launching gray 20k val-selection (gpnn5)"
    $proc2 = Start-Process -FilePath "..\gpnn5.exe" -ArgumentList `
        '--csv','bench_cifar_gray_20k.csv','--input-cols','1024','--output-cols','10',`
        '--loss','bce','--max-epochs','20','--seed','1','--verbose' `
        -RedirectStandardOutput "bg_cifar_gray20k_stdout.txt" `
        -RedirectStandardError "bg_cifar_gray20k_stderr.txt" `
        -WindowStyle Hidden -PassThru
    $proc2.Id | Out-File "bg_cifar_gray20k_pid.txt" -Encoding ascii
    Log "stage3: pid=$($proc2.Id), waiting (max 10h)"
    [void](Wait-ByPidFile "bg_cifar_gray20k_pid.txt" 600)
    Log "stage3 done: final=$(Get-FinalLoss 'bg_cifar_gray20k_stdout.txt') bestval=$(Get-BestVal 'bg_cifar_gray20k_stdout.txt')"
} else {
    Log "stage3 skipped: bench_cifar_gray_20k.csv missing"
}

# ---- Stage 4: report ----
$report = @"
=== CIFAR gray generalization experiments (val-based selection) ===
train-selected (old, 5k, 40ep): 1.91909 train / ~5.1 val (worse than 3.25 base rate)
RGB 5k (train-selected):        1.61764 train
5k  40ep val-selected: train=$(Get-FinalLoss 'bg_cifar_gray_vs_stdout.txt') val=$(Get-BestVal 'bg_cifar_gray_vs_stdout.txt')
20k 20ep val-selected: train=$(Get-FinalLoss 'bg_cifar_gray20k_stdout.txt') val=$(Get-BestVal 'bg_cifar_gray20k_stdout.txt')
val is mean BCE over 10 outputs; base rate = 3.25
"@
$report | Out-File "cifar_gen_report.txt" -Encoding utf8
Log "stage4 done: report written — chain2 complete"
