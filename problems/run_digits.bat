@echo off
REM ============================================================================
REM  Run GP-NN on 8x8 MNIST digits (64 inputs, 10 outputs, BCE)
REM
REM  Prerequisites:
REM    1. Run prepare_digits.py first to generate bench_digits.csv
REM    2. Build gpnn.exe from src/ (g++ -O2 -std=c++17 src/*.cpp -o gpnn.exe
REM       with proper MinGW PATH)
REM
REM  Usage:
REM    cd problems
REM    run_digits.bat
REM
REM  Expected runtime: 5-15 minutes per seed (64-input NEURONs are expensive).
REM  Runs 3 seeds sequentially (the structural search is the bottleneck).
REM ============================================================================

set "PATH=C:\Program Files (x86)\Dev-Cpp\MinGW64\bin;%PATH%"
cd /d "C:\Users\banny\Documents\algorithm project\problems"
set "EXE=..\gpnn.exe"
set "OUT=digits_output"

if not exist %OUT% mkdir %OUT%

echo === Preparing dataset (if not already done) ===
if not exist bench_digits.csv python prepare_digits.py

echo === Running GP-NN on digits (3 seeds) ===

echo [seed 1]
%EXE% --csv bench_digits.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 200 --seed 1 --log-file %OUT%\seed1_log.txt > %OUT%\seed1_stdout.txt 2> %OUT%\seed1_stderr.txt
echo   done

echo [seed 2]
%EXE% --csv bench_digits.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 200 --seed 2 --log-file %OUT%\seed2_log.txt > %OUT%\seed2_stdout.txt 2> %OUT%\seed2_stderr.txt
echo   done

echo [seed 3]
%EXE% --csv bench_digits.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 200 --seed 3 --log-file %OUT%\seed3_log.txt > %OUT%\seed3_stdout.txt 2> %OUT%\seed3_stderr.txt
echo   done

echo === Results ===
echo See %OUT%\seed*_stdout.txt for loss/expression output
echo See baseline_results.txt for sklearn comparison
echo.
echo Quick summary:
for %%s in (1 2 3) do (
  echo seed %%s:
  findstr "Final loss" %OUT%\seed%%s_stdout.txt
  findstr "Structural changes" %OUT%\seed%%s_stdout.txt
  findstr "Expression" %OUT%\seed%%s_stdout.txt
  echo.
)
