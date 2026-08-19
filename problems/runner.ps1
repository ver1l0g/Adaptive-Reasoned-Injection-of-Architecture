# runner.ps1 — self-contained idempotent overnight queue for ARIA.
# Launch: double-click RUN_QUEUE.bat (in the same folder). Do NOT close the
# console window that opens (minimize it). The runner survives independently
# of any editor/session; it dies only on shutdown/logoff.
#
# Idempotent: each stage checks its previous output for a completion marker
# and skips if found. Safe to re-run any number of times.
#
# Writes: QUEUE_LOG.txt (progress), QUEUE_RESULTS.txt (morning report).

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent)
$log = "QUEUE_LOG.txt"
$res = "QUEUE_RESULTS.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }

# ---- AC power check: refuse to start on battery ----
$bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($bat -and $bat.BatteryStatus -ne 2) {
    Write-Host "LAPTOP IS ON BATTERY. Overnight queue will not start." -ForegroundColor Red
    Write-Host "Plug in AC power, then run RUN_QUEUE.bat again."
    Log "ABORTED: on battery at start"
    Read-Host "Press Enter to close"
    exit 1
}

# ---- Keep the machine awake (AC only; closing the lid must not sleep) ----
powercfg /change standby-timeout-ac 0 | Out-Null
powercfg /change hibernate-timeout-ac 0 | Out-Null
# Lid close = do nothing (index 0) on AC
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 | Out-Null
powercfg /setactive SCHEME_CURRENT | Out-Null

Log "=== overnight queue starting (pid $PID) ==="

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
function Launch($exe, $arglist, $tag) {
    $outF = "bg_${tag}_stdout.txt"; $errF = "bg_${tag}_stderr.txt"; $pidF = "bg_${tag}_pid.txt"
    Remove-Item $outF, $errF, $pidF -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $exe -ArgumentList $arglist `
        -RedirectStandardOutput $outF -RedirectStandardError $errF `
        -WindowStyle Hidden -PassThru
    $p.Id | Out-File $pidF -Encoding ascii
}
# Stage done = its stdout contains a terminal marker from a PREVIOUS run.
function StageDone($tag, $marker) {
    $f = "bg_${tag}_stdout.txt"
    if (-not (Test-Path $f)) { return $false }
    return (Select-String -Path $f -Pattern $marker -ErrorAction SilentlyContinue) -ne $null
}
function RunStage($name, $tag, $doneMarker, $exe, $arglist, $maxMin, $resultPattern) {
    if (StageDone $tag $doneMarker) {
        Log "SKIP ${name} (already complete)"
        return
    }
    Log "RUN  ${name}"
    Launch $exe $arglist $tag
    [void](Wait-ByPidFile "bg_${tag}_pid.txt" $maxMin)
    $r = Select-String -Path "bg_${tag}_stdout.txt" -Pattern $resultPattern -ErrorAction SilentlyContinue |
         Select-Object -Last 4
    $rtxt = ($r | ForEach-Object { $_.Line }) -join "`n"
    Log "DONE ${name}:`n${rtxt}`n---"
    ("[$(Get-Date -Format 'HH:mm')] ${name}`n${rtxt}`n") | Out-File $res -Append -Encoding utf8
}

# ========================================================================
# STAGES (resume-safe; already-finished stages auto-skip)
# ========================================================================

# --- Group A: regression gate (d9, t31 were lost when the monitor died) ---
RunStage "reg d9"  "reg_d9_q"  "Evolution Complete" "..\aria6.exe" @(
    '--csv','bench_d9_noise.csv','--input-cols','2',
    '--max-epochs','60','--seed','1','--save-graph','none') 30 @('Final loss')
RunStage "reg t31" "reg_t31_q" "Evolution Complete" "..\aria6.exe" @(
    '--csv','bench_t31_three_way_product.csv','--input-cols','3',
    '--max-epochs','60','--seed','1','--save-graph','none') 30 @('Final loss')

# --- Group B: temporal battery (harness prints narma10 line at end) ---
RunStage "temporal battery" "temporal_q" "narma10" "python" @(
    'run_temporal.py','..\aria6.exe','100') 300 @('lorenz3','lorenz','narma10_lag','narma10')

# --- Group C: Feynman card (harness prints summary at end) ---
Copy-Item "feynman_results.csv" "feynman_results_prequeue.csv" -Force -ErrorAction SilentlyContinue
RunStage "Feynman card (aria6)" "feynman_q" "Feynman subset" "python" @(
    'run_feynman.py','..\aria6.exe','100') 600 @('exact','solved')

# --- Group D: limits battery (7 probes; harness prints firstpos8 last) ---
RunStage "limits battery (aria6)" "limits_q" "firstpos8" "python" @(
    'run_limits.py','..\aria6.exe','150') 420 @('highdim','narma','hetero','stripes','count8','firstpos8')

# --- Group E: char-LM ladder w/ softmax-CE (the language numbers) ---
RunStage "charLM w8" "lmw8_q" "SoftmaxCE" "..\aria6.exe" @(
    '--csv','bench_shakespeare_w8.csv','--input-cols','8','--output-cols','65',
    '--loss','bce','--max-epochs','30','--seed','1','--save-graph','none',
    '--eval-csv','bench_shakespeare_w8.csv') 240 @('SoftmaxCE','Accuracy','Restored best-val')
RunStage "charLM w16" "lmw16_q" "SoftmaxCE" "..\aria6.exe" @(
    '--csv','bench_shakespeare_w16.csv','--input-cols','16','--output-cols','65',
    '--loss','bce','--max-epochs','30','--seed','1','--save-graph','none',
    '--eval-csv','bench_shakespeare_w16.csv') 300 @('SoftmaxCE','Accuracy','Restored best-val')
RunStage "charLM w32" "lmw32_q" "SoftmaxCE" "..\aria6.exe" @(
    '--csv','bench_shakespeare_w32.csv','--input-cols','32','--output-cols','65',
    '--loss','bce','--max-epochs','30','--seed','1','--save-graph','none',
    '--eval-csv','bench_shakespeare_w32.csv') 360 @('SoftmaxCE','Accuracy','Restored best-val')

# --- Group F: I.32.8 with gates (continuing the arc) ---
RunStage "I.32.8 (aria6 gates)" "i328_q" "Held-out Evaluation" "..\aria6.exe" @(
    '--csv','feynman/I.32.8.csv','--input-cols','3',
    '--eval-csv','feynman/I.32.8_test.csv',
    '--max-epochs','150','--seed','1','--save-graph','none') 40 @('Eval R2','Final loss')

Log "=== overnight queue COMPLETE — read QUEUE_RESULTS.txt ==="
("=== QUEUE COMPLETE at $(Get-Date -Format 'HH:mm') ===`n") | Out-File $res -Append -Encoding utf8
Write-Host "ALL DONE. Results: QUEUE_RESULTS.txt. Press Enter to close."
Read-Host | Out-Null
