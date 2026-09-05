# status.ps1 — one command, full picture of every ARIA run.
# Scans known run locations; prints epoch/commit progress, freshness,
# and liveness. Usage: powershell -File harness\status.ps1
Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent | Split-Path)

$now = Get-Date
$live = @{}
Get-Process aria*, gpnn* -ErrorAction SilentlyContinue | ForEach-Object {
    try { $live[$_.Id] = $_.CPU } catch {}
}

function Show-Run($name, $log) {
    if (-not (Test-Path $log)) { return }
    $epochs = (Select-String -Path $log -Pattern 'sgd\s+loss' -ErrorAction SilentlyContinue |
               Measure-Object).Count
    $commits = (Select-String -Path $log -Pattern 'COMMIT rank' -ErrorAction SilentlyContinue |
                Measure-Object).Count
    $done = Select-String -Path $log -Pattern 'Evolution Complete' -Quiet
    $item = Get-Item $log
    $age = [int]($now - $item.LastWriteTime).TotalMinutes
    $eval = (Select-String -Path $log -Pattern 'Eval R2\[out0\]|Eval Accuracy' |
             Select-Object -Last 1).Line
    $evalTxt = if ($eval) { $eval.Trim().Substring(0, [Math]::Min(38, $eval.Trim().Length)) } else { "" }
    $maxEp = if ($log -match 'a25l') { 60 } else { 30 }
    $flag = if ($done) { "DONE" }
            elseif ($age -gt 30) { "STALE ${age}m" }
            else { "live" }
    Write-Host ("{0,-28} ep{1,-4} c{2,-3} {3,-9} {4,3}m  {5}" -f `
        $name, "$epochs", "$commits", $flag, $age, $evalTxt)
}

Write-Host "=== ARIA status $(Get-Date -Format 'HH:mm:ss') | aria procs: $($live.Count) ==="
Write-Host ""

Write-Host "-- language/induction probes --"
Get-ChildItem induction -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    Show-Run $_.Name "$($_.FullName)\run_log.txt"
}
Write-Host "-- poems --"
Get-ChildItem suite-poems -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    Show-Run "poems/$($_.Name)" "$($_.FullName)\run_log.txt"
}
Write-Host "-- arc probes --"
Get-ChildItem arcprobe -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    foreach ($lg in @('log.txt','log2.txt')) {
        if (Test-Path "$($_.FullName)\$lg") { Show-Run "arc/$($_.Name)" "$($_.FullName)\$lg" }
    }
}
Write-Host "-- misc runs (root-level logs) --"
Get-ChildItem hd16_*, hd12_* -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    foreach ($lg in @('log.txt','log2.txt','log3.txt')) {
        if (Test-Path "$($_.FullName)\$lg") { Show-Run $_.Name "$($_.FullName)\$lg" }
    }
}
Get-ChildItem i328ab -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    if (Test-Path "$($_.FullName)\log.txt") { Show-Run "i328/$($_.Name)" "$($_.FullName)\log.txt" }
}
