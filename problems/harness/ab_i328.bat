@echo off
rem ab_i328.bat exe seed tag — I.32.8 A/B run (abort-proof).
setlocal
cd /d "%~dp0.."
set WD=i328ab\%3
if not exist %WD% mkdir %WD%
pushd %WD%
"%~dp0..\..\%1" --csv "..\..\suite-feynman\I.32.8.csv" --input-cols 3 --eval-csv "..\..\suite-feynman\I.32.8_test.csv" --max-epochs 150 --seed %2 --save-graph none > log.txt 2>&1
popd
endlocal
