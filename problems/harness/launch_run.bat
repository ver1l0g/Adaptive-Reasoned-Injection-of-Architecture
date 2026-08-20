@echo off
set "PATH=C:\Program Files (x86)\Dev-Cpp\MinGW64\bin;%PATH%"
cd /d "C:\Users\banny\Documents\algorithm project\problems"
set "EXE=..\gpnn.exe"
set "R=run_output"
if not exist %R% mkdir %R%

start "" /B %EXE% --csv bench_t21_three_region.csv       --input-cols 1  --max-epochs 200 --log-file %R%\t21_log.txt > %R%\t21_stdout.txt 2> %R%\t21_stderr.txt
start "" /B %EXE% --csv bench_t22_windowed_sine.csv      --input-cols 1  --max-epochs 200 --log-file %R%\t22_log.txt > %R%\t22_stdout.txt 2> %R%\t22_stderr.txt
start "" /B %EXE% --csv bench_t23_quadratic.csv          --input-cols 2  --max-epochs 200 --log-file %R%\t23_log.txt > %R%\t23_stdout.txt 2> %R%\t23_stderr.txt
start "" /B %EXE% --csv bench_t24_quadrant_xor.csv       --input-cols 2  --max-epochs 200 --log-file %R%\t24_log.txt > %R%\t24_stdout.txt 2> %R%\t24_stderr.txt
start "" /B %EXE% --csv bench_t31_three_way_product.csv  --input-cols 3  --max-epochs 200 --log-file %R%\t31_log.txt > %R%\t31_stdout.txt 2> %R%\t31_stderr.txt
start "" /B %EXE% --csv bench_t32_sine_of_product.csv    --input-cols 2  --max-epochs 200 --log-file %R%\t32_log.txt > %R%\t32_stdout.txt 2> %R%\t32_stderr.txt
start "" /B %EXE% --csv bench_t33_absolute_value.csv     --input-cols 1  --max-epochs 200 --log-file %R%\t33_log.txt > %R%\t33_stdout.txt 2> %R%\t33_stderr.txt
start "" /B %EXE% --csv bench_t34_xor3_parity.csv        --input-cols 3  --max-epochs 200 --log-file %R%\t34_log.txt > %R%\t34_stdout.txt 2> %R%\t34_stderr.txt
start "" /B %EXE% --csv bench_d1_step.csv                --input-cols 1  --max-epochs 200 --log-file %R%\d1_log.txt  > %R%\d1_stdout.txt 2> %R%\d1_stderr.txt
start "" /B %EXE% --csv bench_d2_mod3.csv                --input-cols 1  --max-epochs 200 --log-file %R%\d2_log.txt  > %R%\d2_stdout.txt 2> %R%\d2_stderr.txt
start "" /B %EXE% --csv bench_d3_two_sines.csv           --input-cols 1  --max-epochs 200 --log-file %R%\d3_log.txt  > %R%\d3_stdout.txt 2> %R%\d3_stderr.txt
start "" /B %EXE% --csv bench_d4_irrelevant12.csv        --input-cols 12 --max-epochs 200 --log-file %R%\d4_log.txt  > %R%\d4_stdout.txt 2> %R%\d4_stderr.txt
start "" /B %EXE% --csv bench_d5_extrap_cliff.csv        --input-cols 1  --max-epochs 200 --sweep 1.0 3.0 0.25 --log-file %R%\d5_log.txt  > %R%\d5_stdout.txt 2> %R%\d5_stderr.txt
start "" /B %EXE% --csv bench_d6_seq_parity.csv          --input-cols 1  --max-epochs 200 --no-shuffle --log-file %R%\d6_log.txt  > %R%\d6_stdout.txt 2> %R%\d6_stderr.txt
start "" /B %EXE% --csv bench_d7_multiout.csv            --input-cols 1  --max-epochs 200 --output-cols 3 --log-file %R%\d7_log.txt  > %R%\d7_stdout.txt 2> %R%\d7_stderr.txt
start "" /B %EXE% --csv bench_d8_compose_absprod.csv     --input-cols 2  --max-epochs 200 --log-file %R%\d8_log.txt  > %R%\d8_stdout.txt 2> %R%\d8_stderr.txt
start "" /B %EXE% --csv bench_d9_noise.csv               --input-cols 2  --max-epochs 200 --log-file %R%\d9_log.txt  > %R%\d9_stdout.txt 2> %R%\d9_stderr.txt
echo LAUNCHED
