@echo off
rem feyn_seed.bat <seed> 鈥?one aria31 feynman seed in an ISOLATED cwd
rem (pool in shared problems/ hangs at startup with 3 concurrent: TBD root
rem cause; isolation fixes the battery). Writes CSV back to results/.
setlocal
cd /d "%~dp0.."
set S=%1
set WD=feyn26\s%S%
if not exist %WD% mkdir %WD%
pushd %WD%
python ..\..\harness\run_feynman.py ..\..\..\aria31.exe 50 %S%
popd
endlocal

