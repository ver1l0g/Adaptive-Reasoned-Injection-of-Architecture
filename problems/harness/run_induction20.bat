@echo off
rem run_induction20.bat — M2.3 attention smoke test (aria20, abort-proof).
setlocal
cd /d "%~dp0.."
set WD=induction\a20
if not exist %WD% mkdir %WD%
pushd %WD%
"%~dp0..\..\aria20.exe" --csv "..\..\induction\s1\train.csv" --input-cols 32 --output-cols 16 --loss bce --max-epochs 30 --seed 1 --save-graph none --eval-csv "..\..\induction\s1\test.csv" > run_log.txt 2>&1
popd
endlocal
