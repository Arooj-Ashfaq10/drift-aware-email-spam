"""Second temporal corpus (comment 39): convert CEAS_08.csv from the Zenodo curated email bundle (record 8339691, the
same bundle used for the Spam-T5 reproduction) into the parsed format idx,label,recv,subject,body used by every runner.
The 'date' column is the sender Date header (the bundle has no Received header), so out-of-range or unparseable dates
are dropped and emails are ordered by date. Report this difference from TREC in the paper.

usage: python prep_ceas.py --csv /kaggle/input/.../CEAS_08.csv --out ceas08_parsed.csv.gz"""
import argparse
import pandas as pd
from src.email_data import build_email_stream, stream_summary

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--periods", type=int, default=12); ap.add_argument("--eval_per_class", type=int, default=50)
ap.add_argument("--max_chars", type=int, default=1500)
a = ap.parse_args()
df = pd.read_csv(a.csv); df.columns = [c.lower() for c in df.columns]
print("columns:", list(df.columns), "| rows:", len(df))
dcol = next(c for c in ("date", "received", "time") if c in df.columns)
df["recv"] = pd.to_datetime(df[dcol], errors="coerce", utc=True)
q = df.recv.quantile([0.01, 0.99]); n0 = len(df)
df = df[df.recv.between(q.iloc[0], q.iloc[1])].sort_values("recv").reset_index(drop=True)
print(f"kept {len(df)} of {n0} rows with a plausible date ({df.recv.min()} .. {df.recv.max()})")
# CEAS-08 covers only about five days and its volume is very uneven within a day, so equal-duration periods leave some
# periods with almost no ham. Keep the date order but space the emails evenly in time: the runners' equal-duration
# periods then become equal-count chunks of the date-ordered stream (stated in the paper).
t0 = df.recv.min(); df["recv"] = t0 + pd.to_timedelta(range(len(df)), unit="m")
print("periods: equal-count chunks of the date-ordered stream")
out = pd.DataFrame(dict(idx=range(len(df)), label=df["label"].astype(int), recv=df.recv.dt.round("s"),
                        subject=df.get("subject", "").fillna("").astype(str).str[:300], body=df["body"].fillna("").astype(str).str[:a.max_chars]))
out.to_csv(a.out, index=False); print("label counts:", out.label.value_counts().to_dict())
s = build_email_stream(a.out, a.periods, a.eval_per_class); print(stream_summary(s).to_string())
