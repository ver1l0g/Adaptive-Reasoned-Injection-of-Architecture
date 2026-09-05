@echo off
rem run_induction23.bat — attention 3x-budget test (aria23, abort-proof).
setlocal
cd /d "%~dp0.."
set WD=induction\a23
if not exist %WD% mkdir %WD%
pushd %WD%
"%~dp0..\..\aria23.exe" --csv "..\..\induction\s1\train.csv" --input-cols 32 --output-cols 16 --loss bce --max-epochs 30 --seed 1 --save-graph none --eval-csv "..\..\induction\s1\test.csv" > run_log.txt 2>&1
popd
endlocal
