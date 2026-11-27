"""Revision analyses (CPU only, from saved predictions; no LLM, no corpus needed).

Covers reviewer comments 20, 32/33, 45-47, 49-50, 52-55:
  * two-level bootstrap CIs (resample seeds, then weeks as blocks)          -> ci_two_level.csv
  * paired bootstrap test of the (1-ROCA)% difference + Holm correction     -> paired_roca_tests.csv
  * McNemar per seed (which run builds each table is explicit)               -> mcnemar_per_seed.csv
  * threshold calibrated on Week-0 data only (no future labels)             -> calibrated.csv
  * spam misclassification at fixed low ham misclassification               -> fixed_hm.csv
  * cumulative label cost vs running (1-ROCA)% per week                      -> cumulative_cost.csv
  * drift magnitude vs benefit of refreshing (Spearman)                      -> drift_vs_benefit.csv
  * monitor threshold sensitivity                                            -> threshold_sensitivity.csv
  * data files for the pgfplots figures                                      -> fig_*.dat

usage: python analyze_revision.py --llm results_email/llm --classical results_email/classical \
           --monitor results_email/monitor --out results_email/revision
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from src.metrics import metrics, mcnemar, one_minus_roca_pct

LLM_POLICIES = None  # derived from the data (every policy not starting with clf_/inc_/bert_)


def make_pairs(names, meta, m_primary="8"):
    """Primary (pre-declared): DTDR(m) vs static, vs the equal-budget even schedule, vs fixed refresh after Week 1,
    vs uncertainty (active) selection at the same budget. Exploratory: the remaining paired comparisons that exist."""
    d = f"dtr_m{m_primary}"; n = int(meta.loc[meta.policy == d, "refreshes"].iloc[0]) if d in set(meta.policy) else None
    cand = [(d, "static"), (d, f"recent_even{n}_m{m_primary}"), (d, f"recent_even{n}"), (d, "fixed_w1"), (d, f"active_even{n}_m{m_primary}")]
    primary = [(x, y) for x, y in cand if x in names and y in names]
    exp = [("static", "zs"), ("recent_R1", "random_R1"), ("recent_R3", "random_R3"), ("recent_R1", "static"), ("random_R1", "static"),
           ("clf_static_b50", "static")]
    exp += [(x, "static") for x in sorted(names) if x.startswith(("dtr_", "randtrig_", "active_", "sel_")) and x != d]
    return primary, [(x, y) for x, y in exp if x in names and y in names and (x, y) not in primary]


def blocks(P):
    """dict policy -> dict seed -> dict week -> (y, score, pred) arrays, drifted weeks 1..11 only."""
    out = {}
    for (pol, s, w), g in P[P.period >= 1].groupby(["policy", "seed", "period"]):
        out.setdefault(pol, {}).setdefault(int(s), {})[int(w)] = (g.y.to_numpy(), g.score.to_numpy(), g.pred.to_numpy())
    return out


def pooled(bl, seeds, weeks, fn):
    vals = []
    for s in seeds:
        ys, ss, ps = zip(*[bl[s][w] for w in weeks])
        vals.append(fn(np.concatenate(ys), np.concatenate(ss), np.concatenate(ps)))
    return float(np.mean(vals))


ROCA = lambda y, s, p: one_minus_roca_pct(y, s)
F1 = lambda y, s, p: 100 * metrics(y, p)["f1"]


def two_level_ci(B, pol, n_boot, rng):
    bl = B[pol]; seeds = sorted(bl); weeks = sorted(bl[seeds[0]]); res = {}
    for name, fn in (("roca", ROCA), ("f1", F1)):
        point = pooled(bl, seeds, weeks, fn); bs = []
        for _ in range(n_boot):
            ss = list(rng.choice(seeds, len(seeds))); ww = list(rng.choice(weeks, len(weeks)))
            bs.append(pooled(bl, ss, ww, fn))
        res[name] = (point, *np.percentile(bs, [2.5, 97.5]))
    return res


def paired_roca(B, a, b, n_boot, rng):
    """Paired bootstrap of mean (1-ROCA)% difference A - B. Seeds are paired by seed id (same evaluation emails);
    a deterministic single-run policy is paired with every seed of the other policy. Weeks are resampled as blocks."""
    sa, sb = sorted(B[a]), sorted(B[b])
    pairs = [(x, x) for x in sa if x in sb] or [(x, y) for x in sa for y in sb]
    weeks = sorted(B[a][sa[0]])
    diff = lambda pp, ww: np.mean([pooled(B[a], [x], ww, ROCA) - pooled(B[b], [y], ww, ROCA) for x, y in pp])
    point = diff(pairs, weeks); bs = []
    for _ in range(n_boot):
        pp = [pairs[i] for i in rng.integers(0, len(pairs), len(pairs))]; ww = list(rng.choice(weeks, len(weeks)))
        bs.append(diff(pp, ww))
    bs = np.array(bs); p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return point, lo, hi, max(p, 1.0 / n_boot)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def best_threshold(y, s):
    """Youden-J threshold (maximises TPR - FPR) on labelled past data."""
    best, tau = -1, 0.0
    for t in np.unique(s):
        p = (s >= t).astype(int); m = metrics(y, p); j = m["recall"] + m["tnr"] - 1
        if j > best: best, tau = j, float(t)
    return tau


def sm_at_hm(y, s, hm_max):
    """Lowest spam misclassification (%) with ham misclassification <= hm_max (%), over all thresholds."""
    y = np.asarray(y); s = np.asarray(s); ham = np.sort(s[y == 0])[::-1]; n_ham = len(ham)
    k = int(np.floor(hm_max / 100 * n_ham))               # ham allowed above threshold
    tau = ham[k] if k < n_ham else -np.inf                 # threshold strictly above the (k+1)-th highest ham score
    return 100 * float((s[y == 1] <= tau).mean()), 100 * k / n_ham


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True); ap.add_argument("--classical", required=True)
    ap.add_argument("--monitor", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--primary_m", default="8"); ap.add_argument("--n_boot", type=int, default=2000); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    P = pd.concat([pd.read_csv(os.path.join(d, "predictions.csv")) for d in (a.llm, a.classical)])
    M = pd.concat([pd.read_csv(os.path.join(d, "run_meta.csv")) for d in (a.llm, a.classical)])
    M.loc[(M.policy == "clf_full") & (M.refreshes == 0), "refreshes"] = 10
    B = blocks(P); pols = sorted(B)

    # 1. two-level bootstrap CIs (comments 44, 45)
    rows = []
    for pol in pols:
        r = two_level_ci(B, pol, a.n_boot, rng)
        rows.append(dict(policy=pol, n_seeds=len(B[pol]), roca=r["roca"][0], roca_lo=r["roca"][1], roca_hi=r["roca"][2],
                         f1=r["f1"][0], f1_lo=r["f1"][1], f1_hi=r["f1"][2]))
    CI = pd.DataFrame(rows).round(3); CI.to_csv(os.path.join(a.out, "ci_two_level.csv"), index=False); print(CI.to_string(), "\n")

    # 2. paired (1-ROCA)% tests, Holm on the 4 pre-declared primary comparisons (comments 43, 46, 47)
    rows = []
    PRIMARY, EXPLORATORY = make_pairs(set(B), M, a.primary_m)
    for fam, pairs in (("primary", PRIMARY), ("exploratory", EXPLORATORY)):
        for x, y in pairs:
            d, lo, hi, p = paired_roca(B, x, y, a.n_boot, rng)
            rows.append(dict(family=fam, A=x, B=y, delta_roca=d, lo=lo, hi=hi, p=p))
    T = pd.DataFrame(rows)
    for fam in ("primary", "exploratory"):
        idx = T.family == fam
        if idx.any(): T.loc[idx, "p_holm"] = holm(T.loc[idx, "p"].to_numpy())
    T = T.round(4); T.to_csv(os.path.join(a.out, "paired_roca_tests.csv"), index=False); print(T.to_string(), "\n")

    # 3. McNemar per seed (comment 46: which run builds each table)
    rows = []
    for x, y in PRIMARY + EXPLORATORY:
        for s in sorted(B[x]):
            sy = s if s in B[y] else sorted(B[y])[0]
            A_ = P[(P.policy == x) & (P.seed == s) & (P.period >= 1)].set_index("id")
            B_ = P[(P.policy == y) & (P.seed == sy) & (P.period >= 1)].set_index("id")
            j = A_.join(B_, lsuffix="_a", rsuffix="_b", how="inner"); b, c, p = mcnemar(j.y_a, j.pred_a, j.pred_b)
            rows.append(dict(A=x, B=y, seed=s, only_A=b, only_B=c, p=round(p, 4)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "mcnemar_per_seed.csv"), index=False)

    # 4. calibrated thresholds from Week-0 labelled data only (comments 32, 33, 49)
    rows = []
    for (pol, s), g in P.groupby(["policy", "seed"]):
        w0, dr = g[g.period == 0], g[g.period >= 1]
        tau = best_threshold(w0.y.to_numpy(), w0.score.to_numpy())
        m0, m1 = metrics(dr.y, (dr.score > 0 if pol.startswith("clf") else dr.score > 0).astype(int)), metrics(dr.y, (dr.score >= tau).astype(int))
        rows.append(dict(policy=pol, seed=s, tau=tau, f1_default=100 * m0["f1"], hm_default=100 * m0["fpr"], sm_default=100 * m0["fnr"],
                         f1_cal=100 * m1["f1"], hm_cal=100 * m1["fpr"], sm_cal=100 * m1["fnr"]))
    C = pd.DataFrame(rows).groupby("policy")[["f1_default", "hm_default", "sm_default", "f1_cal", "hm_cal", "sm_cal"]].agg(["mean", "std"]).round(2)
    C.columns = [f"{x}_{t}" for x, t in C.columns]; C.to_csv(os.path.join(a.out, "calibrated.csv")); print(C.to_string(), "\n")

    # 5. spam misclassification at fixed ham misclassification (comment 50); 550 ham -> 1% = 5 ham, 0.1% unresolvable (=0 ham)
    rows = []
    for (pol, s), g in P[P.period >= 1].groupby(["policy", "seed"]):
        sm1, hm1 = sm_at_hm(g.y, g.score, 1.0); sm0, hm0 = sm_at_hm(g.y, g.score, 0.0)
        rows.append(dict(policy=pol, seed=s, sm_at_hm1=sm1, sm_at_hm0=sm0))
    F = pd.DataFrame(rows).groupby("policy")[["sm_at_hm1", "sm_at_hm0"]].agg(["mean", "std"]).round(2)
    F.columns = [f"{x}_{t}" for x, t in F.columns]; F.to_csv(os.path.join(a.out, "fixed_hm.csv")); print(F.to_string(), "\n")

    # 6. per-week ROCA (3-seed mean), cumulative labels, running ROCA (comment 55)
    PP = pd.DataFrame([dict(policy=pol, seed=s, period=w, roca=one_minus_roca_pct(g.y, g.score), f1=100 * metrics(g.y, g.pred)["f1"])
                       for (pol, s, w), g in P.groupby(["policy", "seed", "period"])])
    W = PP.groupby(["policy", "period"])[["roca", "f1"]].mean().reset_index()
    sched = {r.policy: [int(x) for x in str(r.schedule).split()] if isinstance(r.schedule, str) and r.schedule.strip() else [] for r in M.drop_duplicates("policy").itertuples()}
    per_ref = {"zs": 0}; rows = []
    for pol in [p for p in pols if not p.startswith(("clf_", "inc_", "bert_"))]:
        k = 0 if pol == "zs" else 6; labels = k
        for w in range(1, 12):
            if pol != "zs" and w in sched.get(pol, []): labels += k
            g = P[(P.policy == pol) & P.period.between(1, w)]
            run = np.mean([one_minus_roca_pct(x.y, x.score) for _, x in g.groupby("seed")])
            rows.append(dict(policy=pol, week=w, cum_labels=labels, running_roca=run))
    pd.DataFrame(rows).round(3).to_csv(os.path.join(a.out, "cumulative_cost.csv"), index=False)

    # 7. drift magnitude vs refresh benefit (comment 52): content distance of week t from the demonstrations' source week
    #    vs the gain of dtr/recent refresh over static in week t
    if a.monitor and os.path.exists(os.path.join(a.monitor, "distance_matrix.json")):
        D = np.array(json.load(open(os.path.join(a.monitor, "distance_matrix.json"))))
        if "recent_R1" not in set(W.policy): return
        st = W[W.policy == "static"].set_index("period").roca; r1 = W[W.policy == "recent_R1"].set_index("period").roca
        rows = [dict(week=w, d_from_week0=D[0, w], d_from_prev=D[w - 1, w], static_roca=st[w], gain_recent_R1=st[w] - r1[w]) for w in range(2, 12)]
        DV = pd.DataFrame(rows); rho, p = spearmanr(DV.d_from_prev, DV.gain_recent_R1); rho2, p2 = spearmanr(DV.d_from_week0, DV.static_roca)
        DV.round(4).to_csv(os.path.join(a.out, "drift_vs_benefit.csv"), index=False)
        json.dump(dict(spearman_prevdist_vs_gain=[rho, p], spearman_week0dist_vs_static_roca=[rho2, p2]), open(os.path.join(a.out, "drift_vs_benefit_summary.json"), "w"), indent=1)
        print("drift vs benefit:", round(rho, 3), round(p, 3), "| week0 distance vs static ROCA:", round(rho2, 3), round(p2, 3))

    # 8. figure data (comments 17, 58, 59)
    def dump(name, pols_, col):
        if not pols_: return
        T_ = W[W.policy.isin(pols_)].pivot(index="period", columns="policy", values=col)[pols_]
        T_.round(3).to_csv(os.path.join(a.out, f"fig_{name}.dat"), sep=" ")
    have = set(W.policy); pick = lambda xs: [x for x in xs if x in have]
    ev1 = next((x for x in ("recent_even1_m8", "recent_even1") if x in have), None)
    dump("drift_classical_roca", pick(["clf_static_b50", "clf_static_b500", "clf_full"]), "roca")
    dump("drift_llm_roca", pick(["static", ev1, "dtr_m8"]), "roca")
    dump("f1_llm", pick(["zs", "static", ev1, "recent_R1", "dtr_m8", "dtr_m6"]), "f1")
    dump("f1_classical", pick(["clf_static_b6", "clf_static_b50", "clf_static_b500", "clf_recent_b50", "clf_recent_b500", "clf_full"]), "f1")
    S = CI.merge(M.groupby("policy").labels_used.mean().reset_index(), on="policy")
    S = S.sort_values("labels_used"); best = np.inf; S["pareto"] = 0
    for i, r in S.iterrows():
        if r.roca < best: best = r.roca; S.loc[i, "pareto"] = 1
    S["family"] = np.where(S.policy.str.startswith("clf"), "classical", "llm")
    S.to_csv(os.path.join(a.out, "fig_cost.csv"), index=False); print(S[["policy", "labels_used", "roca", "roca_lo", "roca_hi", "pareto"]].to_string())


if __name__ == "__main__":
    main()
