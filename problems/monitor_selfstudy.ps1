# monitor_selfstudy.ps1 — targeted verification of the two gate fixes.
#   V1: I.32.8 (stack-suppression -> DIVIDE_PRODUCT should win)  ~20 min
#   V2: bigram w1 WITHOUT boolean spam -> softmax-CE number       ~35 min
#   V3: pre-pooled CIFAR to completion (vision answer)            ~45 min
# Then the big battery if still welcome (no deadline this time).
$log = "SELFSTUDY_LOG.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
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
}
Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent)
Log "=== self-study chain armed (aria3) ==="

# V1: I.32.8
Log "V1: I.32.8 with stack-suppression gate"
Launch "..\aria3.exe" @('--csv','feynman/I.32.8.csv','--input-cols','3',
    '--eval-csv','feynman/I.32.8_test.csv',
    '--max-epochs','150','--seed','1','--save-graph','none') "i328_a3"
[void](Wait-ByPidFile "bg_i328_a3_pid.txt" 30)
$r2 = Select-String -Path "bg_i328_a3_stdout.txt" -Pattern "Eval R2\[out0\]:\s*([0-9.eE+-]+)" | Select-Object -Last 1
$cm = (Select-String -Path "bg_i328_a3_stdout.txt" -Pattern "COMMIT.*DIVIDE_PRODUCT" | Measure-Object).Count
Log "V1 done: I.32.8 R2=$($r2.Matches[0].Groups[1].Value) divprod_commits=$cm"

# V2: bigram w1
Log "V2: bigram w1 (aria3, boolean gate)"
Launch "..\aria3.exe" @('--csv','bench_shakespeare_w1.csv','--input-cols','1','--output-cols','65',
    '--loss','bce','--max-epochs','20','--seed','1','--save-graph','none',
    '--eval-csv','bench_shakespeare_w1.csv') "w1_a3"
[void](Wait-ByPidFile "bg_w1_a3_pid.txt" 50)
$sm = Select-String -Path "bg_w1_a3_stdout.txt" -Pattern "SoftmaxCE|Accuracy" | Select-Object -Last 2
$bc = (Select-String -Path "bg_w1_a3_stdout.txt" -Pattern "COMMIT.*BOOLEAN" | Measure-Object).Count
Log "V2 done (boolean_commits=$bc):`n$($sm -join "`n")"

# V3: pooled CIFAR
Log "V3: pre-pooled CIFAR to completion"
Launch "..\aria3.exe" @('--csv','bench_cifar_gray_5k_pooled8.csv','--input-cols','64','--output-cols','10',
    '--loss','bce','--max-epochs','40','--seed','1','--verbose','--save-graph','none') "pooled_a3"
[void](Wait-ByPidFile "bg_pooled_a3_pid.txt" 70)
$vl = Select-String -Path "bg_pooled_a3_stdout.txt" -Pattern "Restored best-val" | Select-Object -Last 1
Log "V3 done: $vl"

Log "=== self-study verification COMPLETE — summary in this log ==="
