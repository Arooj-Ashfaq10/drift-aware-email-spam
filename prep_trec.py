"""Step 1 (CPU): parse the original TREC 2007 corpus into one CSV (idx, label, recv, subject, body)."""
import argparse, os
from src.email_data import parse_corpus, build_email_stream, stream_summary

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="folder that contains data/ and full/index (e.g. /content/trec07p/trec07p)")
ap.add_argument("--out", required=True, help="output CSV(.gz), e.g. /content/drive/MyDrive/trec_work/trec07p_parsed.csv.gz")
ap.add_argument("--limit", type=int, default=None, help="parse only the first N emails (quick test)")
ap.add_argument("--periods", type=int, default=12)
ap.add_argument("--eval_per_class", type=int, default=50)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df = parse_corpus(a.root, a.out, limit=a.limit)
print("\nlabel counts (1 = spam):", df.label.value_counts().to_dict())
print("empty body:", f"{(df.body.str.len() == 0).mean():.1%}", "| empty subject:", f"{(df.subject.str.len() == 0).mean():.1%}")
s = build_email_stream(a.out, a.periods, a.eval_per_class)
print("\nPeriod summary (equal-duration periods over the REAL delivery time):")
print(stream_summary(s).to_string())
