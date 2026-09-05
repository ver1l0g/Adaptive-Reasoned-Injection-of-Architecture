@echo off
rem run_vision_overnight.bat — MNIST digits + CIFAR-10 on aria31 (QoL build:
rem val-budget + quiet). Abort-proof; logs crash-safe. Runs alongside the
rem live attention probe (total 3 processes, within slot guidance).
setlocal
cd /d "%~dp0.."
set EXE=%~dp0..\..\aria31.exe

if not exist vision_out mkdir vision_out

echo [digits] MNIST 8x8, 3 seeds, 200 epochs
"%EXE%" --csv suite-vision\bench_digits.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 200 --seed 1 --save-graph none --quiet > vision_out\digits_s1.txt 2>&1
"%EXE%" --csv suite-vision\bench_digits.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 200 --seed 2 --save-graph none --quiet > vision_out\digits_s2.txt 2>&1
"%EXE%" --csv suite-vision\bench_digits.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 200 --seed 3 --save-graph none --quiet > vision_out\digits_s3.txt 2>&1

echo [cifar-gray-5k] 100 epochs
"%EXE%" --csv suite-vision\bench_cifar_gray_5k.csv --input-cols 1024 --output-cols 10 --loss bce --max-epochs 100 --seed 1 --save-graph none --quiet > vision_out\cifar5k.txt 2>&1

echo [cifar-pooled8] pooled comparison, 100 epochs
"%EXE%" --csv suite-vision\bench_cifar_gray_5k_pooled8.csv --input-cols 1024 --output-cols 10 --loss bce --max-epochs 100 --seed 1 --save-graph none --quiet > vision_out\cifar5k_pooled.txt 2>&1

echo ALL VISION RUNS COMPLETE
endlocal
