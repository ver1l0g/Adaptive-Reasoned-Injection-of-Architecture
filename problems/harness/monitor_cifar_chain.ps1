# monitor_cifar_chain.ps1 — automated CIFAR chain, start-and-forget.
#   Stage 1: wait for running baseline (bg_cifar_pid.txt) to exit
#   Stage 2: launch gray A/B (patch-pooling, gpnn3), wait for exit
#   Stage 3: write cifar_ab_report.txt comparing baseline vs patch-pool
#   Stage 4: launch full RGB 5k run (gpnn3), then finish
$log = "monitor_chain_log.txt"
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

Log "chain armed"

# ---- Stage 1: baseline ----
Log "stage1: waiting for baseline (max 4h)"
[void](Wait-ByPidFile "bg_cifar_pid.txt" 240)
Log "stage1 done: baseline final=$(Get-FinalLoss 'bg_cifar_stdout.txt')"

# ---- Stage 2: gray A/B with patch pooling ----
Log "stage2: launching gray A/B (gpnn3)"
$proc = Start-Process -FilePath "..\gpnn3.exe" -ArgumentList `
    '--csv','bench_cifar_gray_5k.csv','--input-cols','1024','--output-cols','10',`
    '--loss','bce','--max-epochs','40','--seed','1','--verbose' `
    -RedirectStandardOutput "bg_cifar_pp_stdout.txt" `
    -RedirectStandardError "bg_cifar_pp_stderr.txt" `
    -WindowStyle Hidden -PassThru
$proc.Id | Out-File "bg_cifar_pp_pid.txt" -Encoding ascii
Log "stage2: A/B pid=$($proc.Id), waiting (max 5h)"
[void](Wait-ByPidFile "bg_cifar_pp_pid.txt" 300)
Log "stage2 done: A/B final=$(Get-FinalLoss 'bg_cifar_pp_stdout.txt')"

# ---- Stage 3: comparison report ----
$baseLoss = Get-FinalLoss 'bg_cifar_stdout.txt'
$ppLoss   = Get-FinalLoss 'bg_cifar_pp_stdout.txt'
$ppCommits = 0
if (Test-Path "bg_cifar_pp_stdout.txt") {
    $ppCommits = (Select-String -Path "bg_cifar_pp_stdout.txt" -Pattern "COMMIT.*PATCH_POOLING" | Measure-Object).Count
}
$report = @"
=== CIFAR gray 5k: baseline vs patch-pooling A/B ===
baseline (linear only) : $baseLoss
patch-pooling (gpnn3)  : $ppLoss
PATCH_POOLING commits  : $ppCommits
same data/seed (bench_cifar_gray_5k.csv, seed 1)
"@
$report | Out-File "cifar_ab_report.txt" -Encoding utf8
Log "stage3 done: report written"

# ---- Stage 4: full RGB 5k ----
Log "stage4: launching full RGB 5k (gpnn3)"
$proc2 = Start-Process -FilePath "..\gpnn3.exe" -ArgumentList `
    '--csv','bench_cifar_rgb_5k.csv','--input-cols','3072','--output-cols','10',`
    '--loss','bce','--max-epochs','30','--seed','1','--verbose' `
    -RedirectStandardOutput "bg_cifar_rgb_stdout.txt" `
    -RedirectStandardError "bg_cifar_rgb_stderr.txt" `
    -WindowStyle Hidden -PassThru
$proc2.Id | Out-File "bg_cifar_rgb_pid.txt" -Encoding ascii
Log "stage4: RGB pid=$($proc2.Id) — chain complete, monitor exiting"
