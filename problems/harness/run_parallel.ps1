$env:PATH = "C:\Program Files (x86)\Dev-Cpp\MinGW64\bin;" + $env:PATH
$exe = "C:\Users\banny\Documents\algorithm project\gpnn.exe"
$out = "C:\Users\banny\Documents\algorithm project\problems\digits_output"
$wd  = "C:\Users\banny\Documents\algorithm project\problems"

# Clean previous output
Remove-Item $out\* -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $out -Force | Out-Null

$procs = @()
foreach ($s in 1,2,3) {
  $p = Start-Process -FilePath $exe `
    -ArgumentList "--csv","bench_digits.csv","--input-cols","64","--output-cols","10","--loss","bce","--max-epochs","200","--seed","$s","--log-file","$out\seed${s}_log.txt" `
    -WorkingDirectory $wd `
    -RedirectStandardOutput "$out\seed${s}_stdout.txt" `
    -RedirectStandardError  "$out\seed${s}_stderr.txt" `
    -WindowStyle Hidden -PassThru
  $procs += $p
  Write-Output "launched seed $s PID $($p.Id)"
}
Write-Output "waiting for all to finish at $(Get-Date -Format HH:mm:ss)..."
$procs | ForEach-Object { $_.WaitForExit() }
Write-Output "all exited at $(Get-Date -Format HH:mm:ss)"
foreach ($p in $procs) { Write-Output "  PID $($p.Id) exit=$($p.ExitCode)" }
