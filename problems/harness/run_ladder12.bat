@echo off
rem run_ladder12.bat <window> â€?one charLM window on aria12, isolated dir.
rem Usage (from problems/): harness\run_ladder12.bat w8 8
setlocal
cd /d "%~dp0.."
set TAG=%1
set WIN=%2
set WD=ladder12\%TAG%
if not exist %WD% mkdir %WD%
copy /y subgraph_library.txt %WD%\subgraph_library.txt >nul
copy /y failure_library.txt %WD%\failure_library.txt >nul
pushd %WD%
"%~dp0..\..\aria12.exe" --csv ..\..\suite-language\bench_shakespeare_%TAG%.csv --input-cols %WIN% --output-cols 65 --loss bce --max-epochs 30 --seed 1 --save-graph none --eval-csv ..\..\suite-language\bench_shakespeare_%TAG%.csv > run_log.txt 2>&1
popd
endlocal
