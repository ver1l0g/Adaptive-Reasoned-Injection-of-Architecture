@echo off
rem run_pooled8.bat — CIFAR pooled8 rerun (64 inputs; abort-proof wrapper).
setlocal
cd /d "%~dp0.."
if not exist vision_out mkdir vision_out
pushd vision_out
"%~dp0..\..\aria31.exe" --csv "..\suite-vision\bench_cifar_gray_5k_pooled8.csv" --input-cols 64 --output-cols 10 --loss bce --max-epochs 100 --seed 1 --save-graph none --quiet > pooled8_final.txt 2>&1
popd
endlocal
