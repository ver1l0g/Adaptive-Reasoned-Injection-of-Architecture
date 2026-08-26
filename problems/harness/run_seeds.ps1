Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent)
foreach ($s in 2,3,4,5) {
    python harness\run_feynman.py ..\aria10.exe 50 $s 2>&1 | Out-Null
}
"seed battery done" | Out-File multiseed_done.txt
