ARIA OVERNIGHT QUEUE — offline instructions
============================================

BEFORE DISCONNECTING (while you still have internet, optional):
  - none required. Everything needed is on disk.

TO START (in the dorm, no internet needed):
  1. Plug the laptop into AC power (REQUIRED — the queue refuses battery).
  2. Double-click:  RUN_QUEUE.bat   (in the problems folder)
  3. A console window opens. MINIMIZE it (do not close).
     Closing the lid is OK — the runner disables lid-sleep automatically.
  4. Walk away.

IN THE MORNING:
  - Read QUEUE_RESULTS.txt  (one block per stage, plain numbers)
  - QUEUE_LOG.txt has timestamps if something looks missing.
  - If the laptop was interrupted, just double-click RUN_QUEUE.bat again:
    finished stages are skipped automatically.

WHAT IT RUNS (in order, ~8-11 h total):
  1. Regression checks d9 + t31        (~10 min)
  2. Temporal battery (lorenz/narma)   (~30 min)
  3. Feynman card, 25 equations        (~4-5 h)
  4. Limits battery, 7 probes          (~1.5 h)
  5. char-LM w8/w16/w32 softmax-CE     (~1.5-2 h)  <- language numbers
  6. I.32.8 with gate fixes            (~20 min)

TO RESTORE NORMAL POWER SETTINGS afterwards:
  powercfg /change standby-timeout-ac 15
  powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
  powercfg /setactive SCHEME_CURRENT

IF SOMETHING HANGS (rare; one shadow-validation bug known):
  - Task Manager -> end the aria6.exe process. The queue notices within a
    minute, logs it, and moves on to the next stage.
