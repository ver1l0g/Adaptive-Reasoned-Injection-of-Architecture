@echo off
rem run_induction25.bat — attention EVIDENCE-GATE test (aria25, abort-proof).
rem The matrix: a20/a22 (control, 7.007%), a23 (+3x budget), a24 (+seed,
rem 8.21%), a25 (+gate [+budget +seed — all levers live]).
setlocal
cd /d "%~dp0.."
set WD=induction\a25
if not exist %WD% mkdir %WD%
pushd %WD%
"%~dp0..\..\aria25.exe" --csv "..\..\induction\s1\train.csv" --input-cols 32 --output-cols 16 --loss bce --max-epochs 30 --seed 1 --save-graph none --eval-csv "..\..\induction\s1\test.csv" > run_log.txt 2>&1
popd
endlocal
