# finish_feynman.ps1 — run the 5 missing equations, append to the CSV,
# then hand off to the main queue for limits/charLM/I.32.8.
$log = "QUEUE_LOG.txt"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent)

$eqs = @(
    @("II.11.27",3), @("II.34.29a",2), @("III.4.33",2),
    @("III.10.19",2), @("I.48.20",3)
)
foreach ($e in $eqs) {
    $name = $e[0]; $nv = $e[1]
    Log "FEYN-FINISH ${name}"
    $tag = "ffin_$($name -replace '[^a-zA-Z0-9]','_')"
    $outF = "bg_${tag}_stdout.txt"; $errF = "bg_${tag}_stderr.txt"; $pidF = "bg_${tag}_pid.txt"
    Remove-Item $outF, $errF, $pidF -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath "..\aria6.exe" -ArgumentList @(
        '--csv',"feynman/$name.csv",'--input-cols',"$nv",
        '--eval-csv',"feynman/${name}_test.csv",
        '--max-epochs','100','--seed','1','--save-graph','none'
    ) -RedirectStandardOutput $outF -RedirectStandardError $errF -WindowStyle Hidden -PassThru
    $p.Id | Out-File $pidF -Encoding ascii
    $deadline = (Get-Date).AddMinutes(30)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $p.Id -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Seconds 20
    }
    if (Get-Process -Id $p.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $p.Id -Force; Log "FEYN-FINISH ${name} TIMED OUT"
    }
    $r2 = Select-String -Path $outF -Pattern "Eval R2\[out0\]:\s*([0-9.eE+-]+)" -ErrorAction SilentlyContinue | Select-Object -Last 1
    $loss = Select-String -Path $outF -Pattern "Final loss:\s*([0-9.eE+-]+)" -ErrorAction SilentlyContinue | Select-Object -Last 1
    $r2v = if ($r2) { $r2.Matches[0].Groups[1].Value } else { "" }
    $lv = if ($loss) { $loss.Matches[0].Groups[1].Value } else { "" }
    Add-Content "feynman_results.csv" "$name,$nv,$r2v,$lv,0,"
    Log "FEYN-FINISH ${name} done: R2=$r2v"
    ("[$(Get-Date -Format 'HH:mm')] feyn-finish ${name}: R2=$r2v") | Out-File "QUEUE_RESULTS.txt" -Append -Encoding utf8
}
Log "FEYN-FINISH complete — starting main queue for remaining stages"
Start-ScheduledTask -TaskName "ARIA_queue"
Log "main queue resumed (will skip complete stages; feynman stage will re-run and append fresh rows — acceptable overlap, CSV grows monotonically readable)"
