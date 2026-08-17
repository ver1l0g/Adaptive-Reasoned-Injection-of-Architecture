# monitor_offline_chain.ps1 — fully autonomous chain for the offline period.
# No network needed; everything local. Appends progress to OFFLINE_CHAIN_LOG.txt
# and writes a catch-up summary to OFFLINE_RESULTS.txt at each stage boundary.
#
#   Stage 1: wait for the shakespeare harness (w16+w32) to finish
#   Stage 2: rerun w8 (lost earlier) — 30 epochs
#   Stage 3: Feynman rerun with gpnn7 (definitive card; old CSV preserved)
#   Stage 4: Korns rerun with gpnn7 (old CSV preserved)
#   Stage 5: multiseed standard suite, 2 seeds (old CSV preserved)
$log = "OFFLINE_CHAIN_LOG.txt"
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

function Append-Summary($text) { $text | Out-File "OFFLINE_RESULTS.txt" -Append -Encoding utf8 }

Log "offline chain armed"
if (-not (Test-Path "OFFLINE_RESULTS.txt")) {
    "=== OFFLINE PERIOD RESULTS (append-only) ===" | Out-File "OFFLINE_RESULTS.txt" -Encoding utf8
}

# ---- Stage 1: shakespeare harness (already running) ----
Log "stage1: waiting for shakespeare harness (max 6h)"
[void](Wait-ByPidFile "bg_shakespeare_pid.txt" 360)
$sh = Get-Content "bg_shakespeare_stdout.txt" -ErrorAction SilentlyContinue
Append-Summary "`n--- Shakespeare harness (w16, w32) ---`n$sh"
Log "stage1 done"

# ---- Stage 2: w8 rerun ----
Log "stage2: launching w8 rerun (30ep)"
$p = Start-Process -FilePath "..\gpnn7.exe" -ArgumentList `
    '--csv','bench_shakespeare_w8.csv','--input-cols','8','--output-cols','65',`
    '--loss','bce','--max-epochs','30','--seed','1','--save-graph','none' `
    -RedirectStandardOutput "bg_lm_w8_stdout.txt" `
    -RedirectStandardError "bg_lm_w8_stderr.txt" `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File "bg_lm_w8_pid.txt" -Encoding ascii
[void](Wait-ByPidFile "bg_lm_w8_pid.txt" 360)
$val = Select-String -Path "bg_lm_w8_stdout.txt" -Pattern "Restored best-val graph snapshot \(val=([0-9.eE+-]+)" |
       Select-Object -Last 1
$bpc = "?"
if ($val) {
    $v = [double]$val.Matches[0].Groups[1].Value
    $bpc = [math]::Round($v / 0.6931471805599453, 3)
}
Append-Summary "w8 rerun: val_bpc=$bpc bits/char (unigram floor 4.78)"
Log "stage2 done: w8 bpc=$bpc"

# ---- Stage 3: Feynman rerun (preserve old results) ----
if (Test-Path "feynman_results.csv") {
    Copy-Item "feynman_results.csv" "feynman_results_gpnn4era.csv" -Force
}
Log "stage3: launching Feynman rerun (gpnn7)"
$p = Start-Process -FilePath "python" -ArgumentList 'run_feynman.py','..\gpnn7.exe','100' `
    -RedirectStandardOutput "bg_feynman7_stdout.txt" `
    -RedirectStandardError "bg_feynman7_stderr.txt" `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File "bg_feynman7_pid.txt" -Encoding ascii
[void](Wait-ByPidFile "bg_feynman7_pid.txt" 600)
$fs = Get-Content "bg_feynman7_stdout.txt" -ErrorAction SilentlyContinue | Select-Object -Last 4
Append-Summary "`n--- Feynman rerun (gpnn7) ---`n$($fs -join "`n")"
Log "stage3 done"

# ---- Stage 4: Korns rerun ----
if (Test-Path "korns_results.csv") {
    Copy-Item "korns_results.csv" "korns_results_gpnn5era.csv" -Force
}
Log "stage4: launching Korns rerun (gpnn7)"
$p = Start-Process -FilePath "python" -ArgumentList 'run_korns.py','..\gpnn7.exe','200' `
    -RedirectStandardOutput "bg_korns7_stdout.txt" `
    -RedirectStandardError "bg_korns7_stderr.txt" `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File "bg_korns7_pid.txt" -Encoding ascii
[void](Wait-ByPidFile "bg_korns7_pid.txt" 300)
$ks = Get-Content "bg_korns7_stdout.txt" -ErrorAction SilentlyContinue | Select-Object -Last 3
Append-Summary "`n--- Korns rerun (gpnn7, 200ep) ---`n$($ks -join "`n")"
Log "stage4 done"

# ---- Stage 5: multiseed standard suite ----
if (Test-Path "multiseed_results.csv") {
    Copy-Item "multiseed_results.csv" "multiseed_results_prev.csv" -Force
}
Log "stage5: launching multiseed standard suite (17 tasks x 2 seeds, 100ep)"
$p = Start-Process -FilePath "python" -ArgumentList 'run_multiseed.py','--seeds','2',`
    '--epochs','100','--jobs','4','--exe','..\gpnn7.exe' `
    -RedirectStandardOutput "bg_multiseed7_stdout.txt" `
    -RedirectStandardError "bg_multiseed7_stderr.txt" `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File "bg_multiseed7_pid.txt" -Encoding ascii
[void](Wait-ByPidFile "bg_multiseed7_pid.txt" 600)
$ms = Get-Content "bg_multiseed7_stdout.txt" -ErrorAction SilentlyContinue | Select-Object -Last 22
Append-Summary "`n--- Multiseed gpnn7 ---`n$($ms -join "`n")"
Log "stage5 done — offline chain COMPLETE"
Append-Summary "`n=== OFFLINE CHAIN COMPLETE at $(Get-Date -Format 'HH:mm') ==="
