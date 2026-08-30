@echo off
rem run_wujue.bat — the wujue structure-induction run (abort-proof wrapper).
setlocal
cd /d "%~dp0.."
set WD=suite-poems\run2
if not exist %WD% mkdir %WD%
pushd %WD%
"%~dp0..\..\aria18.exe" --csv "..\bench_wujue_train.csv" --input-cols 19 --output-cols 501 --loss bce --max-epochs 30 --seed 1 --save-graph none --eval-csv "..\bench_wujue_test.csv" > run_log.txt 2>&1
popd
endlocal
