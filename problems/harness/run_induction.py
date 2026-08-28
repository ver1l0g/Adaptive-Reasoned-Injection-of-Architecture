#!/usr/bin/env python3
"""M2.4 attention decision probe: induction/copy task generator + harness.

The task (induction heads smoke test): a random sequence of tokens from a
small vocabulary, ending with a QUERY marker; the target is the token that
followed the PREVIOUS occurrence of the queried token. Solving it requires
attending across context — exactly what a windowed-BCE/EMBED trunk cannot
do (fixed-depth mixing) and what attention provides.

Metric: next-token accuracy on the query slot, chance = 1/V.
Ladder point: w32 + EMBED trunk. Decision rule (ROADMAP M2.4):
  - w32 bits/char stalls above ~3.8 AND probe accuracy ~ chance
    -> attention justified (M2.3 go)
  - EMBED keeps scaling or probe solved                 -> defer again

Usage (from problems/):
  python harness/run_induction.py ..\\aria14.exe            # 1 run, seed 1
  python harness/run_induction.py ..\\aria14.exe 5          # 5 seeds
"""
import csv, math, os, re, subprocess, sys, time

EXE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "..\\aria14.exe"
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1

V = 16          # vocabulary (tokens 0..13 usable, 14=SEP, 15=QUERY)
W = 32          # context width (matches the ladder's widest point)
SEQ_LEN = 24    # tokens before the query block
N_TRAIN = 8000
N_TEST = 1000

def gen_split(path, n, seed):
    import random
    rng = random.Random(seed)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # header: in_0..in_{W-1}, out_0..out_{V-1} (one-hot answer)
        w.writerow([f"in_{i}" for i in range(W)] + [f"out_{j}" for j in range(V)])
        rows = 0
        while rows < n:
            seq = [rng.randrange(0, V - 2) for _ in range(SEQ_LEN)]
            # query a token that occurs at least twice (guarantee a unique
            # answer: take the last occurrence with a successor)
            positions = {}
            for i, t in enumerate(seq):
                positions.setdefault(t, []).append(i)
            cands = [t for t, ps in positions.items() if len(ps) >= 2]
            if not cands:
                continue
            q = rng.choice(cands)
            q_pos = positions[q][-2]      # previous-to-last occurrence
            answer = seq[q_pos + 1] if q_pos + 1 < len(seq) else seq[0]
            # window = seq + [SEP, q] padded/truncated to W from the LEFT
            ctx = seq + [V - 2, q]
            ctx = ctx[-W:]
            ctx = [0] * (W - len(ctx)) + ctx
            onehot = [0] * V
            onehot[answer] = 1
            w.writerow(ctx + onehot)
            rows += 1

ACC_RE = re.compile(r"Eval Accuracy:\s*([0-9.]+)%")
BPC_RE = re.compile(r"Eval SoftmaxCE:\s*([0-9.]+)")

def run_one(seed):
    wd = os.path.join("induction", f"s{seed}")
    os.makedirs(wd, exist_ok=True)
    train_p = os.path.abspath(os.path.join(wd, "train.csv"))
    test_p = os.path.abspath(os.path.join(wd, "test.csv"))
    gen_split(train_p, N_TRAIN, seed * 1000 + 1)
    gen_split(test_p, N_TEST, seed * 1000 + 2)
    args = [EXE, "--csv", train_p, "--input-cols", str(W),
            "--output-cols", str(V), "--loss", "bce",
            "--max-epochs", "30", "--seed", str(seed),
            "--save-graph", "none", "--eval-csv", test_p]
    log = os.path.join(wd, "run_log.txt")
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, cwd=wd,
                       timeout=14400)
    text = open(log, encoding="utf-8", errors="replace").read()
    am = ACC_RE.search(text)
    bm = BPC_RE.search(text)
    acc = float(am.group(1)) if am else None
    bpc = (float(bm.group(1)) / math.log(2)) if bm else None
    print(f"seed {seed}: acc={acc if acc is None else round(acc, 2)}% "
          f"(chance {100.0 / V:.1f}%)  eval={bpc if bpc is None else round(bpc, 3)} nats->bpc "
          f"({(time.time() - t0) / 60:.0f}min)", flush=True)

for s in range(1, SEEDS + 1):
    run_one(s)
