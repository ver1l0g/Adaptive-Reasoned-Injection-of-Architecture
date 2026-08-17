#!/usr/bin/env python3
"""Feynman-equation subset benchmark for GP-NN (SRBench-style).

25 curated equations from the Feynman lectures (Udrescu AI-Feynman set),
spanning 1-4 variables and operators: + - * / ^2 ^3 sqrt sin cos and
nested compositions. Noise-free (canonical Feynman track).

Writes per equation:
  feynman/<id>.csv        4000 train rows
  feynman/<id>_test.csv   1000 held-out rows
"""
import csv, math, os, random

# (id, formula, fn, [(lo,hi) per var])
EQS = [
    ("I.6.2",    "p=m*v",                  lambda x: x[0]*x[1],                     [(1,5),(1,5)]),
    ("I.6.2b",   "rho=m/V",                lambda x: x[0]/x[1],                     [(1,5),(1,5)]),
    ("I.7.9",    "t=sqrt(2x/a)",           lambda x: math.sqrt(2*x[0]/x[1]),        [(1,5),(1,5)]),
    ("I.8.4",    "KE=mv^2/2",              lambda x: x[0]*x[1]**2/2,                [(1,5),(1,5)]),
    ("I.9.5",    "F=GmM/r^2",              lambda x: x[0]*x[1]/x[2]**2,             [(1,5),(1,5),(1,5)]),
    ("I.10.7",   "m=m0/sqrt(1-v2^2)",      lambda x: x[0]/math.sqrt(1-x[1]**2),     [(1,5),(0,0.9)]),
    ("I.12.5",   "U=-q1q2/r",              lambda x: -x[0]*x[1]/x[2],               [(1,5),(1,5),(1,5)]),
    ("I.13.12",  "E=q^2/2C",               lambda x: x[0]**2*x[1]**2/2,             [(1,5),(1,5)]),
    ("I.14.3",   "xcm=(m1x1+m2x2)/(m1+m2)",lambda x:(x[0]*x[1]+x[2]*x[3])/(x[0]+x[2]),[(1,5),(1,5),(1,5),(1,5)]),
    ("I.15.3",   "w=(u+v)/(1+uv)",         lambda x:(x[0]-x[1])/(1-x[0]*x[1]),      [(0.1,0.9),(0.1,0.9)]),
    ("I.15.10",  "w=(u+v)/(1+uv)",         lambda x:(x[0]+x[1])/(1+x[0]*x[1]),      [(0.1,0.9),(0.1,0.9)]),
    ("I.18.4",   "E=F/q",                  lambda x: x[0]*x[1]/x[2],                [(1,5),(1,5),(1,5)]),
    ("I.18.14",  "F=qvB",                  lambda x: x[0]*x[1]*x[2],                [(1,5),(1,5),(1,5)]),
    ("I.24.6",   "pr=p0*sqrt(1-v^2)",      lambda x: x[0]*math.sqrt(1-x[1]**2),     [(1,5),(0,0.95)]),
    ("I.25.9",   "f=d1d2/(d1+d2)",         lambda x: x[0]*x[1]/(x[0]+x[1]),         [(1,5),(1,5)]),
    ("I.29.16",  "y=A*sin(kx)",            lambda x: x[0]*math.sin(x[1]*x[2]),      [(1,3),(0.5,3),(0.5,3)]),
    ("I.32.8",   "P=q2a2/(c3)",            lambda x: x[0]**2*x[1]**2/x[2]**3,       [(1,4),(1,4),(1,4)]),
    ("I.34.6",   "w=sqrt((1-v)/(1+v))",    lambda x: math.sqrt((1-x[0])/(1+x[0])),  [(0,0.9)]),
    ("I.43.27",  "lam=kT/(pd2)",           lambda x: x[0]/(x[1]**2*x[2]),           [(1,4),(1,4),(1,4)]),
    ("I.47.23",  "v=sqrt(T/mu)",           lambda x: math.sqrt(x[0]/x[1]),          [(1,5),(1,5)]),
    ("II.11.27", "flux=EAcos",             lambda x: x[0]*x[1]*math.cos(x[2]),      [(1,5),(1,5),(0,1.5)]),
    ("II.34.29a","u=epsE^2/2",             lambda x: x[0]*x[1]**2/2,                [(1,5),(1,5)]),
    ("III.4.33", "E=p^2/2m",               lambda x: x[0]**2/(2*x[1]),              [(1,5),(1,5)]),
    ("III.10.19","pc=sqrt(px2+py2)",       lambda x: math.sqrt(x[0]**2+x[1]**2),    [(1,5),(1,5)]),
    ("I.48.20",  "E=sqrt((pc)^2+(mc2)^2)", lambda x: math.sqrt((x[0]*x[1])**2+x[2]**2), [(1,4),(1,4),(1,4)]),
]

def main(seed=42):
    os.makedirs("feynman", exist_ok=True)
    rng = random.Random(seed)
    for eid, name, fn, ranges in EQS:
        k = len(ranges)
        header = [f"x{i}" for i in range(k)] + ["y"]
        with open(f"feynman/{eid}.csv", "w", newline="") as ftr, \
             open(f"feynman/{eid}_test.csv", "w", newline="") as fte:
            wtr, wte = csv.writer(ftr), csv.writer(fte)
            wtr.writerow(header); wte.writerow(header)
            for row_idx in range(1500):
                xs = [rng.uniform(lo, hi) for (lo, hi) in ranges]
                y = fn(xs)
                row = [f"{v:.6f}" for v in xs] + [f"{y:.6f}"]
                (wtr if row_idx < 1000 else wte).writerow(row)
    print(f"Wrote {len(EQS)} equations -> feynman/ (1000 train + 500 test each)")

if __name__ == "__main__":
    main()
