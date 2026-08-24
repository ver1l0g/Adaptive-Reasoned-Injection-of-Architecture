# monitor_sceladder.ps1 — charLM ladder with softmax-CE on aria8 (new binary).
# The language experiment, properly equipped: SCE loss + shadow subsample
# speedup + library guards. w1 -> w8 -> w16 -> w32, 10k subsets (fast,
# ranking-equivalent) with full-set eval.
$log = "SCELADDER_LOG.txt"
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
function RunStage($name, $tag, $exe, $arglist, $maxMin) {
    Log "RUN ${name}"
    Launch $exe $arglist $tag
    [void](Wait-ByPidFile "bg_${tag}_pid.txt" $maxMin)
    $sm = Select-String -Path "bg_${tag}_stdout.txt" -Pattern "SoftmaxCE|Accuracy|Restored best-val" -ErrorAction SilentlyContinue |
          Select-Object -Last 3
    Log "DONE ${name}:`n$($sm -join "`n")"
    ("[$(Get-Date -Format 'HH:mm')] ${name}`n$($sm -join "`n")`n") | Out-File "SCELADDER_RESULTS.txt" -Append -Encoding utf8
}
Set-Location (Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent)
Log "=== SCE ladder armed (aria8) ==="

# w1 (10k subset exists)
RunStage "w1-SCE" "sce_w1" "..\aria9.exe" @(
    '--csv','suite-language/bench_shakespeare_w1_10k.csv','--input-cols','1','--output-cols','65',
    '--loss','sce','--max-epochs','25','--seed','1','--save-graph','none',
    '--eval-csv','suite-language/bench_shakespeare_w1_10k.csv') 90

# w8 (subset of the 60k file — make 10k first)
python -c "import csv,random; rows=list(csv.reader(open('suite-language/bench_shakespeare_w8.csv'))); h,d=rows[0],rows[1:]; r=random.Random(11); idx=r.sample(range(len(d)),10000); w=csv.writer(open('suite-language/bench_shakespeare_w8_10k.csv','w',newline='')); w.writerow(h); [w.writerow(d[i]) for i in idx]" 2>$null
RunStage "w8-SCE" "sce_w8" "..\aria9.exe" @(
    '--csv','suite-language/bench_shakespeare_w8_10k.csv','--input-cols','8','--output-cols','65',
    '--loss','sce','--max-epochs','25','--seed','1','--save-graph','none',
    '--eval-csv','suite-language/bench_shakespeare_w8_10k.csv') 120

# w16
python -c "import csv,random; rows=list(csv.reader(open('suite-language/bench_shakespeare_w16.csv'))); h,d=rows[0],rows[1:]; r=random.Random(11); idx=r.sample(range(len(d)),10000); w=csv.writer(open('suite-language/bench_shakespeare_w16_10k.csv','w',newline='')); w.writerow(h); [w.writerow(d[i]) for i in idx]" 2>$null
RunStage "w16-SCE" "sce_w16" "..\aria9.exe" @(
    '--csv','suite-language/bench_shakespeare_w16_10k.csv','--input-cols','16','--output-cols','65',
    '--loss','sce','--max-epochs','25','--seed','1','--save-graph','none',
    '--eval-csv','suite-language/bench_shakespeare_w16_10k.csv') 150

# w32
python -c "import csv,random; rows=list(csv.reader(open('suite-language/bench_shakespeare_w32.csv'))); h,d=rows[0],rows[1:]; r=random.Random(11); idx=r.sample(range(len(d)),10000); w=csv.writer(open('suite-language/bench_shakespeare_w32_10k.csv','w',newline='')); w.writerow(h); [w.writerow(d[i]) for i in idx]" 2>$null
RunStage "w32-SCE" "sce_w32" "..\aria9.exe" @(
    '--csv','suite-language/bench_shakespeare_w32_10k.csv','--input-cols','32','--output-cols','65',
    '--loss','sce','--max-epochs','25','--seed','1','--save-graph','none',
    '--eval-csv','suite-language/bench_shakespeare_w32_10k.csv') 180

Log "=== SCE ladder COMPLETE — see SCELADDER_RESULTS.txt ==="
