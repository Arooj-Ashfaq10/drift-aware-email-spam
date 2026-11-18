"""Step 2 (CPU): label-free content-drift monitor -> which periods would DTR refresh at, for several multipliers."""
import argparse, json, os
import numpy as np
from src.email_data import build_email_stream
from src.strategies import period_stats, dtr_schedule, even_schedule

ap = argparse.ArgumentParser()
ap.add_argument("--parsed", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--periods", type=int, default=12)
ap.add_argument("--eval_per_class", type=int, default=50)
ap.add_argument("--mults", type=float, nargs="+", default=[2, 3, 4, 6, 8, 16])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
s = build_email_stream(a.parsed, a.periods, a.eval_per_class)
c, nz = period_stats(s); T = len(c)
print("sampling-noise floor per period (split-half content distance):", [round(nz[p], 4) for p in range(T)])
cons = [1 - float(c[p - 1] @ c[p]) for p in range(1, T)]
print("consecutive-period content distance d(p-1,p):", [round(x, 4) for x in cons])
print("distance from period 0 to p:", [round(1 - float(c[0] @ c[p]), 4) for p in range(1, T)])
print("full distance matrix d(i,j) (rows i, columns j):")
for i in range(T): print(i, [round(1 - float(c[i] @ c[j]), 3) for j in range(T)])
res = {}
for m in a.mults:
    sch, traj = dtr_schedule(c, nz, m)
    res[str(m)] = dict(refresh_periods=sch, n=len(sch))
    print(f"mult={m:g} -> DTR refreshes before periods {sch} (n={len(sch)}) | equal-budget even control: {even_schedule(T, len(sch))}")
json.dump(res, open(os.path.join(a.out, "dtr_schedules.json"), "w"), indent=2)
