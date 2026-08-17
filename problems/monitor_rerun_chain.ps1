# monitor_rerun_chain.ps1 — definitive rerun chain (crash-safe harnesses).
#   Stage 1: Feynman rerun (gpnn7, 100ep)   ~3.5h
#   Stage 2: Korns rerun (gpnn7, 200ep)     ~1h
#   Stage 3: multiseed suite x2 seeds       ~3h
#   Stage 4: char-LM w8 -> w16 -> w32       ~6-8h
# All harnesses now tee subprocess stdout to per-run log files, so every
# number survives kills/timeouts/crashes.
$log = "RERUN_CHAIN_LOG.txt"
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

function Run-Stage($name, $exe, $arglist, $tag, $maxMin, $tailLines) {
    Log "launching ${name}"
    $outF = "bg_${tag}_stdout.txt"; $errF = "bg_${tag}_stderr.txt"; $pidF = "bg_${tag}_pid.txt"
    Remove-Item $outF, $errF, $pidF -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $exe -ArgumentList $arglist `
        -RedirectStandardOutput $outF -RedirectStandardError $errF `
        -WindowStyle Hidden -PassThru
    $p.Id | Out-File $pidF -Encoding ascii
    [void](Wait-ByPidFile $pidF $maxMin)
    $t = Get-Content $outF -ErrorAction SilentlyContinue | Select-Object -Last $tailLines
    Log "${name} done:`n$($t -join "`n")"
}

Log "=== rerun chain armed ==="

# Stage 1: Feynman (old card already preserved)
if (Test-Path "feynman_results.csv") {
    Copy-Item "feynman_results.csv" "feynman_results_shutdown_partial.csv" -Force
}
Run-Stage "Feynman" "python" @('run_feynman.py','..\gpnn7.exe','100') "feynman_r" 420 6

# Stage 2: Korns
if (Test-Path "korns_results.csv") {
    Copy-Item "korns_results.csv" "korns_results_prev.csv" -Force
}
Run-Stage "Korns" "python" @('run_korns.py','..\gpnn7.exe','200') "korns_r" 240 12

# Stage 3: multiseed
if (Test-Path "multiseed_results.csv") {
    Copy-Item "multiseed_results.csv" "multiseed_results_prev2.csv" -Force
}
Run-Stage "Multiseed" "python" @('run_multiseed.py','--seeds','2','--epochs','100','--jobs','4','--exe','..\gpnn7.exe') "multiseed_r" 600 24

# Stage 4: char-LM (the language ceiling question, w-scaling)
Run-Stage "CharLM" "python" @('run_shakespeare.py','..\gpnn7.exe','30') "shakespeare_r" 900 5

Log "=== rerun chain COMPLETE ==="
