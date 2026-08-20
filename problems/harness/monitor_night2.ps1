# monitor_night2.ps1 — overnight battery (aria6, gate fixes + full verification).
#   S1: NARMA-30 with memory-signature gate (MULTI_TAP should finally commit)
#   S2: stripes20 with IFELSE fatigue
#   S3: regression gate — d6, narma10, t24, d9, t31 (memory + core suite)
#   S4: temporal battery (lorenz/narma10 family, full re-verify)
#   S5: Feynman card with aria6 (gate-fix impact on all 25)
#   S6: limits battery re-run (all 7 probes with gates)
#   S7: summary -> NIGHT2_RESULTS.txt
$log = "NIGHT2_LOG.txt"
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
function Launch($exe, $arglist, $tag) {
    $outF = "bg_${tag}_stdout.txt"; $errF = "bg_${tag}_stderr.txt"; $pidF = "bg_${tag}_pid.txt"
    Remove-Item $outF, $errF, $pidF -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $exe -ArgumentList $arglist `
        -RedirectStandardOutput $outF -RedirectStandardError $errF `
        -WindowStyle Hidden -PassThru
    $p.Id | Out-File $pidF -Encoding ascii
}
function RunSingle($name, $exe, $arglist, $tag, $maxMin, $patterns) {
    Log "S: ${name}"
    Launch $exe $arglist $tag
    [void](Wait-ByPidFile "bg_${tag}_pid.txt" $maxMin)
    $res = Select-String -Path "bg_${tag}_stdout.txt" -Pattern $patterns -ErrorAction SilentlyContinue |
           Select-Object -Last 4
    Log "${name} done:`n$($res -join "`n")"
}
Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent)
Log "=== night2 battery armed (aria6) ==="

# S1: NARMA-30
RunSingle "NARMA-30 (memory gate)" "..\aria6.exe" @(
    '--csv','bench_narma30.csv','--input-cols','1','--no-shuffle',
    '--max-epochs','150','--seed','1','--save-graph','none',
    '--eval-csv','bench_narma30.csv') "n30_n2" 40 @('Eval R2','COMMIT.*MULTI_TAP','Final loss')

# S2: stripes20
RunSingle "stripes20 (IFELSE fatigue)" "..\aria6.exe" @(
    '--csv','bench_stripes20.csv','--input-cols','1',
    '--max-epochs','150','--seed','1','--save-graph','none',
    '--eval-csv','bench_stripes20.csv') "str_n2" 40 @('Eval R2','Final loss','COMMIT')

# S3: regression gate (sequential, quick)
foreach ($t in @(
    @('d6','bench_d6_seq_parity.csv','1','--no-shuffle'),
    @('narma10','bench_narma10.csv','1','--no-shuffle'),
    @('t24','bench_t24_quadrant_xor.csv','2',''),
    @('d9','bench_d9_noise.csv','2',''),
    @('t31','bench_t31_three_way_product.csv','3',''))) {
    $extra = @(); if ($t[3]) { $extra = $t[3] }
    $al = @('--csv',$t[1],'--input-cols',$t[2]) + $extra + @('--max-epochs','60','--seed','1','--save-graph','none')
    RunSingle "reg $($t[0])" "..\aria6.exe" $al "reg_$($t[0])_n2" 30 @('Final loss')
}

# S4: temporal battery
Log "S4: temporal battery"
Launch "python" @('run_temporal.py','..\aria6.exe','100') "temporal_n2"
[void](Wait-ByPidFile "bg_temporal_n2_pid.txt" 300)
Log "temporal:`n$((Get-Content bg_temporal_n2_stdout.txt -ErrorAction SilentlyContinue) -join "`n")"

# S5: Feynman card
Copy-Item "feynman_results.csv" "feynman_results_night1.csv" -Force -ErrorAction SilentlyContinue
Log "S5: Feynman card (aria6)"
Launch "python" @('run_feynman.py','..\aria6.exe','100') "feynman_n2"
[void](Wait-ByPidFile "bg_feynman_n2_pid.txt" 600)
Log "feynman:`n$((Get-Content bg_feynman_n2_stdout.txt -ErrorAction SilentlyContinue | Select-Object -Last 6) -join "`n")"

# S6: limits battery
Log "S6: limits battery (aria6)"
Launch "python" @('run_limits.py','..\aria6.exe','150') "limits_n2"
[void](Wait-ByPidFile "bg_limits_n2_pid.txt" 420)
Log "limits:`n$((Get-Content bg_limits_n2_stdout.txt -ErrorAction SilentlyContinue) -join "`n")"

# S7: summary
$summary = @"
=== NIGHT2 RESULTS (aria6: gate fixes) ===
$(Get-Content $log | Select-Object -Skip 1 | Out-String)
"@
$summary | Out-File "NIGHT2_RESULTS.txt" -Encoding utf8
Log "=== night2 COMPLETE ==="
