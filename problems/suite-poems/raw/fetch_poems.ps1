# fetch_poems.ps1 — retry the sparse checkout until the network cooperates.
# The partial clone has the tree; only blob fetches fail intermittently.
Set-Location "C:\Users\banny\Documents\algorithm project\problems\suite-poems\raw\cprepo"
$tang = [string][char]0x5168 + [char]0x5510 + [char]0x8BD7
for ($i = 1; $i -le 30; $i++) {
    $n = (Get-ChildItem $tang -Filter *.json -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($n -ge 1000) {
        "SUCCESS: $n json files" | Out-File fetch_status.txt -Encoding utf8
        exit 0
    }
    Remove-Item .git\index.lock -Force -ErrorAction SilentlyContinue
    git checkout 2>&1 | Out-Null
    Start-Sleep -Seconds 60
}
"FAILED after 30 attempts" | Out-File fetch_status.txt -Encoding utf8
