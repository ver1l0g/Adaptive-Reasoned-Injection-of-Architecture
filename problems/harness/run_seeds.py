import subprocess, sys
for s in [2, 3, 4, 5]:
    print(f'=== seed {s} ===', flush=True)
    subprocess.run([sys.executable, 'harness/run_feynman.py', '..\\\\aria10.exe', '50', str(s)])
print('battery done')
