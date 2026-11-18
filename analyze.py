"""Step 6 (CPU): metrics, significance tests, confidence intervals and figures from the saved predictions."""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.metrics import metrics, mcnemar, bootstrap_ci, one_minus_roca_pct, bootstrap_ci_roca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="result folders (each with predictions.csv + run_meta.csv)")
    ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    P = pd.concat([pd.read_csv(os.path.join(d, "predictions.csv")) for d in a.dirs]); M = pd.concat([pd.read_csv(os.path.join(d, "run_meta.csv")) for d in a.dirs])
    rows = []
    for (pol, s), g in P.groupby(["policy", "seed"]):
        dr, st = g[g.period >= 1], g[g.period == 0]; m, ms = metrics(dr.y, dr.pred), metrics(st.y, st.pred)
        rows.append(dict(policy=pol, seed=s, acc=100 * m["acc"], f1=100 * m["f1"], hm=100 * m["fpr"], sm=100 * m["fnr"], roca=one_minus_roca_pct(dr.y, dr.score),
                         stable_f1=100 * ms["f1"], stable_hm=100 * ms["fpr"], stable_sm=100 * ms["fnr"], stable_roca=one_minus_roca_pct(st.y, st.score)))
    S = pd.DataFrame(rows).merge(M[["policy", "seed", "refreshes", "labels_used"]], on=["policy", "seed"]); S.to_csv(os.path.join(a.out, "summary_by_seed.csv"), index=False)
    cols = ["acc", "f1", "hm", "sm", "roca", "stable_f1", "stable_roca", "refreshes", "labels_used"]
    agg = S.groupby("policy")[cols].agg(["mean", "std"]); agg.columns = [f"{c}_{t}" for c, t in agg.columns]; agg["n_seeds"] = S.groupby("policy").size()
    agg.round(3).to_csv(os.path.join(a.out, "summary.csv")); print(agg[["f1_mean", "hm_mean", "sm_mean", "roca_mean", "stable_roca_mean", "labels_used_mean", "n_seeds"]].round(2).to_string())
    pp = [dict(policy=pol, seed=s, period=t, f1=100 * metrics(g.y, g.pred)["f1"], hm=100 * metrics(g.y, g.pred)["fpr"], sm=100 * metrics(g.y, g.pred)["fnr"], roca=one_minus_roca_pct(g.y, g.score))
          for (pol, s, t), g in P.groupby(["policy", "seed", "period"])]
    PP = pd.DataFrame(pp); PP.to_csv(os.path.join(a.out, "per_period.csv"), index=False)
    # ---- paired McNemar tests (seed 0, drifted periods) ----
    names = set(P.policy); refs = M[M.seed == 0].set_index("policy").refreshes.to_dict(); pairs = []
    if {"static", "zs"} <= names: pairs.append(("static", "zs"))
    pairs += [(x, "static") for x in sorted(names) if x not in ("zs", "static") and not x.startswith("clf_") and "static" in names]
    for x in sorted(names):
        if x.startswith("recent_R") and "random_" + x[7:] in names: pairs.append((x, "random_" + x[7:]))
        if x.startswith("dtr_m") and f"recent_even{refs.get(x)}" in names: pairs.append((x, f"recent_even{refs[x]}"))
    tests = []
    for x, y_ in pairs:
        A = P[(P.policy == x) & (P.seed == 0) & (P.period >= 1)].set_index("id"); B = P[(P.policy == y_) & (P.seed == 0) & (P.period >= 1)].set_index("id")
        j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        if len(j):
            b, c, p = mcnemar(j.y_a, j.pred_a, j.pred_b); tests.append(dict(A=x, B=y_, only_A_correct=b, only_B_correct=c, p=round(p, 4), n=len(j)))
    T = pd.DataFrame(tests); T.to_csv(os.path.join(a.out, "mcnemar_tests.csv"), index=False); print("\n", T.to_string() if len(T) else "no pairs")
    ci = []
    for pol in sorted(names):
        g = P[(P.policy == pol) & (P.seed == 0) & (P.period >= 1)]
        if len(g):
            f_lo, f_hi = bootstrap_ci(g.y, g.pred, "f1", 500); r_lo, r_hi = bootstrap_ci_roca(g.y, g.score, 300)
            ci.append(dict(policy=pol, f1_ci_low=100 * f_lo, f1_ci_high=100 * f_hi, roca_ci_low=r_lo, roca_ci_high=r_hi))
    pd.DataFrame(ci).round(3).to_csv(os.path.join(a.out, "confidence_intervals.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for pol, g in PP.groupby("policy"):
        m = g.groupby("period").f1.mean(); ax.plot(m.index, m.values, marker="o", ms=2.5, lw=1, label=pol)
    ax.set_xlabel("Period (0 = stable window)"); ax.set_ylabel("F1 (%)"); ax.grid(alpha=.3); ax.legend(fontsize=5.5, ncol=3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "f1_by_period.png"), dpi=200); plt.close()
    fig, ax = plt.subplots(figsize=(5.4, 3.6)); A_ = S.groupby("policy")[["labels_used", "roca"]].mean()
    ax.scatter(A_.labels_used, A_.roca)
    for pol, r in A_.iterrows(): ax.annotate(pol, (r.labels_used, r.roca), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xscale("symlog"); ax.set_xlabel("Labelled examples used"); ax.set_ylabel("(1 - ROCA)%  [lower is better]"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "cost_vs_roca.png"), dpi=200); plt.close()
    print("saved to", a.out)


if __name__ == "__main__":
    main()
