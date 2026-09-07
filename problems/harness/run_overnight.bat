@echo off
rem run_overnight.bat ??vision + language battery on aria-cert (abort-proof).
rem Each line = one run, sequentially inside this bat; log per run.
setlocal
cd /d "%~dp0.."
set EXE=%~dp0..\..\aria-cert.exe
if not exist vision_out mkdir vision_out
if not exist lang_out mkdir lang_out

echo [digits s1] %date% %time%
"%EXE%" --csv suite-vision\bench_digits.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 200 --seed 1 --save-graph none --quiet > scratch\runs\vision_out\cert_digits_s1.txt 2>&1
echo [digits s2] %date% %time%
"%EXE%" --csv suite-vision\bench_digits.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 200 --seed 2 --save-graph none --quiet > scratch\runs\vision_out\cert_digits_s2.txt 2>&1
echo [digits s3] %date% %time%
"%EXE%" --csv suite-vision\bench_digits.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 200 --seed 3 --save-graph none --quiet > scratch\runs\vision_out\cert_digits_s3.txt 2>&1
echo [cifar gray-5k] %date% %time%
"%EXE%" --csv suite-vision\bench_cifar_gray_5k.csv --input-cols 1024 --output-cols 10 --loss bce --max-epochs 100 --seed 1 --save-graph none --quiet > scratch\runs\vision_out\cert_cifar5k.txt 2>&1
echo [cifar pooled8] %date% %time%
"%EXE%" --csv suite-vision\bench_cifar_gray_5k_pooled8.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 100 --seed 1 --save-graph none --quiet > scratch\runs\vision_out\cert_cifar5k_pooled.txt 2>&1
echo [cifar rgb-5k] %date% %time%
"%EXE%" --csv suite-vision\bench_cifar_rgb_5k.csv --input-cols 3072 --output-cols 10 --loss bce --max-epochs 100 --seed 1 --save-graph none --quiet > scratch\runs\vision_out\cert_cifar_rgb.txt 2>&1

echo [w1] %date% %time%
"%EXE%" --csv suite-language\bench_shakespeare_w1.csv --input-cols 1 --output-cols 65 --loss bce --max-epochs 30 --seed 1 --save-graph none > scratch\runs\lang_out\cert_w1.txt 2>&1
echo [w8] %date% %time%
"%EXE%" --csv suite-language\bench_shakespeare_w8.csv --input-cols 8 --output-cols 65 --loss bce --max-epochs 30 --seed 1 --save-graph none > scratch\runs\lang_out\cert_w8.txt 2>&1
echo [w16] %date% %time%
"%EXE%" --csv suite-language\bench_shakespeare_w16.csv --input-cols 16 --output-cols 65 --loss bce --max-epochs 30 --seed 1 --save-graph none > scratch\runs\lang_out\cert_w16.txt 2>&1
echo [w32] %date% %time%
"%EXE%" --csv suite-language\bench_shakespeare_w32.csv --input-cols 32 --output-cols 65 --loss bce --max-epochs 30 --seed 1 --save-graph none > scratch\runs\lang_out\cert_w32.txt 2>&1

echo [wujue poems] %date% %time%
rem aria32 for poems: needs the one-hot code-guard + expansion (post-freeze;
rem aria-cert predates them ??codes would be z-scored and the gate reject)
"%~dp0..\..\aria32.exe" --csv suite-poems\bench_wujue_train.csv --input-cols 19 --output-cols 501 --loss bce --max-epochs 30 --seed 1 --save-graph none --eval-csv suite-poems\bench_wujue_test.csv > scratch\runs\lang_out\cert_wujue.txt 2>&1

echo OVERNIGHT BATTERY COMPLETE %date% %time%
endlocal



