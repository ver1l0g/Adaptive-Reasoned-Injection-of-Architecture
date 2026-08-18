# monitor_classrun.ps1 — 85-minute bounded battery (aria2, gain-init build).
# Priority-ordered for the time window; each stage's results are banked
# incrementally. Hard deadline: stages stop launching after 85 min.
$log = "CLASSRUN_LOG.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
$t0 = Get-Date
function TimeLeft { param() [int](($t0.AddMinutes(85) - (Get-Date)).TotalMinutes) }

function Wait-ByPidFile($pidFile, $maxMinutes) {
    $deadline = (Get-Date).AddMinutes($maxMinutes)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path $pidFile) {
            $p = Get-Content $pidFile -ErrorAction SilentlyContinue
            if ($p -and -not (Get-Process -Id $p -ErrorAction SilentlyContinue)) { return $true }
        }
        Start-Sleep -Seconds 30
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
    return $p.Id
}

Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent)
Log "=== class-run armed (85 min budget) ==="

# ---- P1: I.32.8 gain-init verification (~15 min, the headline) ----
Log "P1: I.32.8 with gain-init (aria2)"
[void](Launch "..\aria2.exe" @(
    '--csv','feynman/I.32.8.csv','--input-cols','3',
    '--eval-csv','feynman/I.32.8_test.csv',
    '--max-epochs','120','--seed','1','--save-graph','none') "i328_a2")
[void](Wait-ByPidFile "bg_i328_a2_pid.txt" 20)
$r2 = Select-String -Path "bg_i328_a2_stdout.txt" -Pattern "Eval R2\[out0\]:\s*([0-9.eE+-]+)" | Select-Object -Last 1
$cm = (Select-String -Path "bg_i328_a2_stdout.txt" -Pattern "COMMIT.*DIVIDE_PRODUCT" | Measure-Object).Count
Log "P1 done: I.32.8 R2=$($r2.Matches[0].Groups[1].Value) divprod_commits=$cm  (time left: $(TimeLeft) min)"

# ---- P2: bigram w1 with softmax-CE (~25 min, language floor question) ----
if (TimeLeft -gt 35) {
    Log "P2: bigram w1 (aria2 + softmax-CE eval)"
    [void](Launch "..\aria2.exe" @(
        '--csv','bench_shakespeare_w1.csv','--input-cols','1','--output-cols','65',
        '--loss','bce','--max-epochs','20','--seed','1','--save-graph','none',
        '--eval-csv','bench_shakespeare_w1.csv') "w1_a2")
    [void](Wait-ByPidFile "bg_w1_a2_pid.txt" 35)
    $sm = Select-String -Path "bg_w1_a2_stdout.txt" -Pattern "SoftmaxCE|Accuracy" | Select-Object -Last 2
    Log "P2 done:`n$($sm -join "`n")  (time left: $(TimeLeft) min)"
}

# ---- P3: pre-pooled CIFAR (~40 min, vision prior question) ----
if (TimeLeft -gt 25) {
    Log "P3: pre-pooled CIFAR 8x8 (aria2)"
    [void](Launch "..\aria2.exe" @(
        '--csv','bench_cifar_gray_5k_pooled8.csv','--input-cols','64','--output-cols','10',
        '--loss','bce','--max-epochs','25','--seed','1','--verbose','--save-graph','none') "pooled_a2")
    [void](Wait-ByPidFile "bg_pooled_a2_pid.txt" ([Math]::Min(45, (TimeLeft))))
    $vl = Select-String -Path "bg_pooled_a2_stdout.txt" -Pattern "Restored best-val|^\s*\[\s*\d+\]" | Select-Object -Last 2
    Log "P3 partial: $($vl -join ' | ')  (time left: $(TimeLeft) min)"
}

# ---- P4: temporal battery if time remains (~20 min) ----
if (TimeLeft -gt 25) {
    Log "P4: temporal battery (aria2)"
    [void](Launch "python" @('run_temporal.py','..\aria2.exe','100') "temporal_a2")
    [void](Wait-ByPidFile "bg_temporal_a2_pid.txt" ([Math]::Min(25, (TimeLeft))))
    $tp = Get-Content "bg_temporal_a2_stdout.txt" -ErrorAction SilentlyContinue | Select-Object -Last 4
    Log "P4 partial:`n$($tp -join "`n")"
}

Log "=== class-run complete (stopping before deadline; kill any stragglers manually if needed) ==="
