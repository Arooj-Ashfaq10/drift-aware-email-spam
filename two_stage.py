"""Step 8 (CPU, comment 49): two-stage filter from saved predictions. A cheap classifier (clf_static_b50, 50 Week-0 labels)
decides the emails it is confident about; only the least confident fraction q goes to the frozen LLM (static prompt,
6 demonstrations drawn from the same Week-0 labels, so 50 labels in total). Both thresholds (Youden) and the
confidence band are set on Week 0 only; results are on Weeks 1-11, per seed (seed s of the LLM paired with seed s
of the classifier), then averaged.

usage: python two_stage.py --llm main10/predictions.csv --clf classical/predictions.csv --out results_email/two_stage"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

ap = argparse.ArgumentParser()
ap.add_argument("--llm", required=True); ap.add_argument("--clf", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--llm_policy", default="static"); ap.add_argument("--clf_policy", default="clf_static_b50")
ap.add_argument("--fracs", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.5, 1.0])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
L = pd.read_csv(a.llm); L = L[L.policy == a.llm_policy][["seed", "period", "id", "y", "score"]].rename(columns={"score": "s_llm"})
C = pd.read_csv(a.clf); C = C[C.policy == a.clf_policy][["seed", "period", "id", "score"]].rename(columns={"score": "s_clf"})
D = L.merge(C, on=["seed", "period", "id"]); print("merged rows:", len(D), "| seeds:", sorted(D.seed.unique()))


def youden(y, s):
    f, t, th = roc_curve(y, s); return th[np.argmax(t - f)]


rows = []
for seed, g in D.groupby("seed"):
    w0, ev = g[g.period == 0], g[g.period >= 1]
    tc, tl = youden(w0.y, w0.s_clf), youden(w0.y, w0.s_llm)
    conf0 = (w0.s_clf - tc).abs(); conf = (ev.s_clf - tc).abs()
    for q in a.fracs:
        band = np.quantile(conf0, q) if 0 < q < 1 else (-1 if q == 0 else np.inf)
        to_llm = conf <= band if q < 1 else np.ones(len(ev), bool)
        pred = np.where(to_llm, ev.s_llm > tl, ev.s_clf > tc).astype(int); y = ev.y.values
        rows.append(dict(seed=seed, frac_week0=q, sent_to_llm=100 * to_llm.mean(),
                         hm=100 * ((pred == 1) & (y == 0)).sum() / (y == 0).sum(),
                         sm=100 * ((pred == 0) & (y == 1)).sum() / (y == 1).sum(),
                         err=100 * (pred != y).mean()))
R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out, "two_stage_per_seed.csv"), index=False)
S = R.groupby("frac_week0").agg(sent_to_llm=("sent_to_llm", "mean"), hm=("hm", "mean"), hm_sd=("hm", "std"),
                                sm=("sm", "mean"), sm_sd=("sm", "std"), err=("err", "mean"), err_sd=("err", "std")).round(2)
S.to_csv(os.path.join(a.out, "two_stage.csv")); print(S.to_string())
print("frac 0 = classifier alone, frac 1 = LLM alone (both with Week-0 Youden thresholds)")
