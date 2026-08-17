# monitor_overnight.ps1 — full autonomous overnight sequence (aria2 build).
#   Stage 0: wait for in-flight runs (chain CharLM, gpnn9 Feynman, diag chain)
#   Stage 1: I.32.8 verification (gain-init fix) — quick
#   Stage 2: full Feynman card with aria2 (the definitive one)
#   Stage 3: Korns card with aria2 (200ep)
#   Stage 4: temporal benchmarks with aria2
#   Stage 5: multiseed suite with aria2 (regression gate for gain-init)
#   Stage 6: char-LM ladder w1/w8/w16/w32 with aria2 + softmax-CE eval
#   Stage 7: diagnostics (pre-pooled CIFAR w/ softmax-CE)
#   Stage 8: final summary -> OVERNIGHT_RESULTS.txt
$log = "OVERNIGHT_LOG.txt"
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
function Run-Stage($name, $arglist, $tag, $maxMin, $tail) {
    Log "launching ${name}"
    $outF = "bg_${tag}_stdout.txt"; $errF = "bg_${tag}_stderr.txt"; $pidF = "bg_${tag}_pid.txt"
    Remove-Item $outF, $errF, $pidF -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath "python" -ArgumentList $arglist `
        -RedirectStandardOutput $outF -RedirectStandardError $errF `
        -WorkingDirectory (Get-Location) -WindowStyle Hidden -PassThru
    $p.Id | Out-File $pidF -Encoding ascii
    [void](Wait-ByPidFile $pidF $maxMin)
    $t = Get-Content $outF -ErrorAction SilentlyContinue | Select-Object -Last $tail
    Log "${name} done:`n$($t -join "`n")"
    $t
}

Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent)
Log "=== overnight chain armed (aria2) ==="

# ---- Stage 0: wait for in-flight ----
Log "stage0: waiting for rerun chain (CharLM is its last stage)"
[void](Wait-ByPidFile "bg_shakespeare_r_pid.txt" 900)
Log "stage0: rerun chain done; waiting gpnn9 Feynman"
[void](Wait-ByPidFile "bg_feynman9_pid.txt" 600)
Log "stage0: gpnn9 Feynman done; waiting diag chain"
$deadline = (Get-Date).AddMinutes(600)
while ((Get-Date) -lt $deadline) {
    $d = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
         Where-Object { $_.CommandLine -like "*monitor_diag_chain*" }
    if (-not $d) { break }
    Start-Sleep -Seconds 60
}
Log "stage0 complete — machine free"

# ---- Stage 1: I.32.8 verification ----
Log "stage1: I.32.8 gain-init verification"
Remove-Item bg_i328_stdout.txt, bg_i328_stderr.txt, bg_i328_pid.txt -ErrorAction SilentlyContinue
$p = Start-Process -FilePath "..\aria2.exe" -ArgumentList `
    '--csv','feynman/I.32.8.csv','--input-cols','3',`
    '--eval-csv','feynman/I.32.8_test.csv','--max-epochs','200','--seed','1','--save-graph','none' `
    -RedirectStandardOutput "bg_i328_stdout.txt" -RedirectStandardError "bg_i328_stderr.txt" `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File "bg_i328_pid.txt" -Encoding ascii
[void](Wait-ByPidFile "bg_i328_pid.txt" 240)
$r2 = Select-String -Path "bg_i328_stdout.txt" -Pattern "Eval R2\[out0\]:\s*([0-9.eE+-]+)" | Select-Object -Last 1
$cm = (Select-String -Path "bg_i328_stdout.txt" -Pattern "COMMIT.*DIVIDE_PRODUCT" | Measure-Object).Count
Log "stage1 done: I.32.8 R2=$($r2.Matches[0].Groups[1].Value) divprod_commits=$cm"

# ---- Stage 2: definitive Feynman card ----
Copy-Item "feynman_results.csv" "feynman_results_gpnn9era.csv" -Force -ErrorAction SilentlyContinue
Run-Stage "Feynman (aria2)" @('run_feynman.py','..\aria2.exe','100') "feynman_a2" 480 8 | Out-Null

# ---- Stage 3: Korns ----
Copy-Item "korns_results.csv" "korns_results_chain.csv" -Force -ErrorAction SilentlyContinue
Run-Stage "Korns (aria2)" @('run_korns.py','..\aria2.exe','200') "korns_a2" 300 12 | Out-Null

# ---- Stage 4: temporal ----
Run-Stage "Temporal (aria2)" @('run_temporal.py','..\aria2.exe','200') "temporal_a2" 240 6 | Out-Null

# ---- Stage 5: multiseed (gain-init regression gate) ----
Run-Stage "Multiseed (aria2)" @('run_multiseed.py','--seeds','2','--epochs','100','--jobs','4','--exe','..\aria2.exe') "multiseed_a2" 600 24 | Out-Null

# ---- Stage 6: char-LM ladder ----
Run-Stage "CharLM ladder (aria2)" @('run_shakespeare.py','..\aria2.exe','30') "shakespeare_a2" 1200 5 | Out-Null

# ---- Stage 7: pre-pooled CIFAR diagnostic ----
Log "stage7: pre-pooled CIFAR (aria2)"
Remove-Item bg_pooled_a2_stdout.txt, bg_pooled_a2_stderr.txt, bg_pooled_a2_pid.txt -ErrorAction SilentlyContinue
$p = Start-Process -FilePath "..\aria2.exe" -ArgumentList `
    '--csv','bench_cifar_gray_5k_pooled8.csv','--input-cols','64','--output-cols','10',`
    '--loss','bce','--max-epochs','40','--seed','1','--verbose','--save-graph','none' `
    -RedirectStandardOutput "bg_pooled_a2_stdout.txt" -RedirectStandardError "bg_pooled_a2_stderr.txt" `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File "bg_pooled_a2_pid.txt" -Encoding ascii
[void](Wait-ByPidFile "bg_pooled_a2_pid.txt" 480)
$vl = Select-String -Path "bg_pooled_a2_stdout.txt" -Pattern "Restored best-val" | Select-Object -Last 1
Log "stage7 done: $vl"

# ---- Stage 8: summary ----
$summary = @"
=== OVERNIGHT RESULTS (aria2: gain-init + all fixes) ===
$(Get-Content OVERNIGHT_LOG.txt | Select-Object -Skip 1 | Out-String)
=== raw per-suite outputs ===
Feynman: see feynman_results.csv / bg_feynman_a2_stdout.txt
Korns:   see bg_korns_a2_stdout.txt
Temporal: see bg_temporal_a2_stdout.txt
Multiseed: see bg_multiseed_a2_stdout.txt
CharLM:  see bg_shakespeare_a2_stdout.txt
I.32.8:  see bg_i328_stdout.txt
Pooled CIFAR: see bg_pooled_a2_stdout.txt
"@
$summary | Out-File "OVERNIGHT_RESULTS.txt" -Encoding utf8
Log "=== overnight chain COMPLETE ==="
