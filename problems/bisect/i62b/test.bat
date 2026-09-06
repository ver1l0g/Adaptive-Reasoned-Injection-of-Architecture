@echo off
cd /d "%~dp0"
%1 --csv "..\..\suite-feynman\I.6.2b.csv" --input-cols 2 --eval-csv "..\..\suite-feynman\I.6.2b_test.csv" --max-epochs 50 --seed 1 --save-graph none > %2.txt 2>&1
