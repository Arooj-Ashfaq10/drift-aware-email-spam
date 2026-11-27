"""Step 7 (CPU, comment 28): does a shorter monitoring window shorten the one-period refresh delay?
Re-runs the label-free monitor on TREC 2007 with 12 (weekly), 24 (half-week) and 48 (quarter-week) periods and reports
when the first alarm would trigger a refresh, measured in days after the documented Week 0 -> Week 1 change, plus the
number of further alarms. No labels and no LLM are used, so this measures the monitor only.

usage: python window_size.py --parsed trec07p_parsed.csv.gz --out results_email/window_size"""
import argparse, json, os
import pandas as pd
from src.email_data import build_email_stream
from src.strategies import period_stats, dtr_schedule

ap = argparse.ArgumentParser()
ap.add_argument("--parsed", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--periods", type=int, nargs="+", default=[12, 24, 48]); ap.add_argument("--mults", type=float, nargs="+", default=[6, 8])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rows = []
for P in a.periods:
    s = build_email_stream(a.parsed, P, eval_per_class=5)
    span = (s.recv.max() - s.recv.min()).total_seconds() / 86400; L = span / P; change = span / 12   # change at end of Week 0
    c, nz = period_stats(s)
    for m in a.mults:
        sch, traj = dtr_schedule(c, nz, m)
        first = sch[0] if sch else None
        rows.append(dict(periods=P, days_per_period=round(L, 2), mult=m, refreshes=len(sch), refresh_before=sch,
                         first_refresh_day=round(first * L, 1) if first else None,
                         delay_after_change_days=round(first * L - change, 1) if first else None,
                         alarms_before_change=sum(1 for r in sch if (r - 1) * L < change - 1e-9),
                         first_post_change_delay_days=next((round(r * L - change, 1) for r in sch if (r - 1) * L >= change - 1e-9), None),
                         labels=6 + 6 * len(sch)))
        print(rows[-1])
df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "window_size.csv"), index=False)
print(df.drop(columns="refresh_before").to_string(index=False))
