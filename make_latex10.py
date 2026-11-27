"""Turn the 10-seed analysis outputs into LaTeX blocks for the RM paper (tables + pgfplots data), black and white only.

usage: python make_latex10.py --main ../../kaggle_runs/main10 --classical ../../kaggle_runs/cpu_out/out/classical
          --rev ../../kaggle_runs/rev10 --synthetic ../../kaggle_runs/cpu_out/out/synthetic --out ../../kaggle_runs/latex10"""
import argparse, json, os
import numpy as np, pandas as pd
from src.metrics import metrics, one_minus_roca_pct
from analyze_revision import blocks, paired_roca, holm

BS = chr(92)
def tt(x): return x.replace("_", BS + "_")
def pm(m, s, d=2): return f"{m:.{d}f}$" + BS + f"pm${s:.{d}f}"

LLM_MAIN = ["zs", "static", "recent_R1", "random_R1", "recent_R3", "random_R3", "fixed_w1", "recent_even1_m8", "recent_even2_m6", "dtr_m8", "dtr_m6"]
CLF_MAIN = ["clf_static_b6", "clf_static_b50", "clf_static_b500", "clf_recent_b6", "clf_recent_b50", "clf_recent_b500",
            "inc_sgd_b6", "inc_sgd_b50", "inc_sgd_b500", "inc_pa_b50", "inc_pa_b500", "clf_full"]
ABL = ["dtr_m8", "randtrig_m8", "dtr_noreset_m8", "dtr_guard_m8", "active_dtr_m8", "active_even1_m8", "recent_even1_m8",
       "dtr_m6", "randtrig_m6", "dtr_noreset_m6", "dtr_guard_m6", "active_dtr_m6", "active_even2_m6", "recent_even2_m6"]
NAME = {"recent_even1_m8": "recent\\_even1", "recent_even2_m6": "recent\\_even2"}
def nm(p): return NAME.get(p, tt(p))


def main():
    ap = argparse.ArgumentParser()
    for k in ("main", "classical", "rev", "synthetic", "out"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); W = lambda n, t: open(os.path.join(a.out, n), "w", encoding="utf-8").write(t)
    P = pd.concat([pd.read_csv(os.path.join(d, "predictions.csv")) for d in (a.main, a.classical)])
    M = pd.concat([pd.read_csv(os.path.join(d, "run_meta.csv")) for d in (a.main, a.classical)])
    M.loc[(M.policy == "clf_full") & (M.refreshes == 0), "refreshes"] = 10
    D = P[P.period >= 1]
    rows = []
    for (pol, s), g in D.groupby(["policy", "seed"]):
        m = metrics(g.y, g.pred); rows.append(dict(policy=pol, seed=s, acc=100 * m["acc"], f1=100 * m["f1"], hm=100 * m["fpr"], sm=100 * m["fnr"],
                                                     roca=one_minus_roca_pct(g.y, g.score)))
    R = pd.DataFrame(rows); mean = R.groupby("policy").mean(numeric_only=True); std = R.groupby("policy").std(numeric_only=True).fillna(0)
    meta = M.groupby("policy")[["refreshes", "labels_used"]].mean()
    if "labels_inspected" in M: meta = meta.join(M.groupby("policy").labels_inspected.mean())
    nseed = R.groupby("policy").seed.nunique()
    json.dump(dict(mean=mean.round(3).to_dict(), std=std.round(3).to_dict()), open(os.path.join(a.out, "per_policy.json"), "w"), indent=1)

    # --- main table
    def row(p):
        bold = p.startswith("dtr_m")
        lab = meta.loc[p]; ins = lab.get("labels_inspected", np.nan)
        lab_s = f"{lab.labels_used:,.0f}".replace(",", "{,}") + ("" if pd.isna(ins) else f" / {ins:.0f}")
        single = nseed[p] == 1
        cells = [f"{mean.loc[p, c]:.2f}" if single else pm(mean.loc[p, c], std.loc[p, c]) for c in ("acc", "f1", "hm", "sm", "roca")]
        n = (BS + "textbf{" + nm(p) + "}") if bold else nm(p)
        return " & ".join([n] + cells + [f"{lab.refreshes:.0f}", lab_s]) + BS + BS
    body = [row(p) for p in LLM_MAIN if p in mean.index] + [BS + "midrule"] + [row(p) for p in CLF_MAIN if p in mean.index]
    W("tab_main.tex", "\n".join(body))

    # --- CI table
    CI = pd.read_csv(os.path.join(a.rev, "ci_two_level.csv")).set_index("policy")
    body = [f"{nm(p)} & {CI.loc[p,'f1']:.2f} [{CI.loc[p,'f1_lo']:.2f}, {CI.loc[p,'f1_hi']:.2f}] & {CI.loc[p,'roca']:.3f} [{CI.loc[p,'roca_lo']:.3f}, {CI.loc[p,'roca_hi']:.3f}]" + BS + BS
            for p in LLM_MAIN + CLF_MAIN if p in CI.index]
    W("tab_ci.tex", "\n".join(body))

    # --- paired tests (analysis output + extra label-matched/label-heavy comparisons)
    T = pd.read_csv(os.path.join(a.rev, "paired_roca_tests.csv"))
    B = blocks(P); rng = np.random.default_rng(1); extra = []
    for x, y in (("dtr_m6", "recent_even2_m6"), ("dtr_m8", "recent_R1"), ("dtr_m6", "recent_R1"), ("dtr_m8", "randtrig_m8"), ("dtr_m8", "dtr_noreset_m8"),
                 ("dtr_m8", "dtr_guard_m8"), ("dtr_m6", "randtrig_m6"), ("inc_sgd_b50", "static")):
        if x in B and y in B:
            d, lo, hi, p = paired_roca(B, x, y, 1000, rng); extra.append(dict(family="secondary", A=x, B=y, delta_roca=d, lo=lo, hi=hi, p=p))
    E = pd.DataFrame(extra); E["p_holm"] = holm(E.p.to_numpy()); T = pd.concat([T, E]); T.to_csv(os.path.join(a.out, "paired_all.csv"), index=False)
    def fp(p): return "$<0.001$" if p < 0.001 else f"{p:.3f}"
    lines = []
    for fam, title in (("primary", "Primary (pre-declared, Holm over 4)"), ("secondary", "Ablation and budget comparisons (Holm over 8)"),
                       ("exploratory", "Exploratory, selected (Holm over all exploratory)")):
        t = T[T.family == fam]
        if fam == "exploratory": t = t[t.A.isin(["static", "recent_R1", "random_R1", "clf_static_b50"])]
        if t.empty: continue
        lines.append(BS + "multicolumn{5}{l}{" + BS + "emph{" + title + "}}" + BS + BS)
        for r in t.itertuples():
            lines.append(f"{nm(r.A)} vs {nm(r.B)} & {r.delta_roca:+.2f} & [{r.lo:+.2f}, {r.hi:+.2f}] & {fp(r.p)} & {fp(r.p_holm)}" + BS + BS)
    W("tab_paired.tex", "\n".join(lines))

    # --- McNemar aggregated over the 10 seeds
    Mc = pd.read_csv(os.path.join(a.rev, "mcnemar_per_seed.csv"))
    keep = [("static", "zs"), ("dtr_m8", "static"), ("dtr_m6", "static"), ("dtr_m8", "recent_even1_m8"), ("dtr_m8", "active_even1_m8"),
            ("recent_R1", "static"), ("random_R1", "static"), ("recent_R1", "random_R1")]
    lines = []
    for x, y in keep:
        g = Mc[(Mc.A == x) & (Mc.B == y)]
        if g.empty: continue
        wa = int(((g.p < 0.05) & (g.only_A > g.only_B)).sum()); wb = int(((g.p < 0.05) & (g.only_B > g.only_A)).sum())
        lines.append(f"{nm(x)} & {nm(y)} & {g.only_A.sum()} & {g.only_B.sum()} & {wa} / {wb} / {len(g) - wa - wb}" + BS + BS)
    W("tab_mcnemar.tex", "\n".join(lines))

    # --- calibration / fixed hm
    C = pd.read_csv(os.path.join(a.rev, "calibrated.csv")).set_index("policy"); F = pd.read_csv(os.path.join(a.rev, "fixed_hm.csv")).set_index("policy")
    sel = ["zs", "static", "recent_R1", "recent_even1_m8", "dtr_m8", "dtr_m6", "clf_static_b50", "inc_sgd_b50", "clf_static_b500", "clf_recent_b500", "clf_full"]
    lines = [f"{nm(p)} & " + " & ".join(f"{C.loc[p, c]:.1f}" for c in ("f1_default_mean", "hm_default_mean", "sm_default_mean", "f1_cal_mean", "hm_cal_mean", "sm_cal_mean"))
             + f" & {F.loc[p,'sm_at_hm1_mean']:.1f} & {F.loc[p,'sm_at_hm0_mean']:.1f}" + BS + BS for p in sel if p in C.index]
    W("tab_calib.tex", "\n".join(lines))

    # --- ablation table
    lines = []
    for p in ABL:
        if p not in mean.index: continue
        lab = meta.loc[p]; tp = T[(T.A == p) & (T.B == "static")]
        d = "--" if p == "static" or tp.empty else f"{tp.delta_roca.iloc[0]:+.2f}"
        sch = M[M.policy == p].schedule.astype(str).iloc[0].replace("nan", "--")
        lines.append(f"{nm(p)} & {pm(mean.loc[p,'roca'], std.loc[p,'roca'])} & {lab.labels_used:.0f} / {lab.get('labels_inspected', 0):.0f} & {d}" + BS + BS)
        if p == "recent_even1_m8": lines.append(BS + "midrule")
    W("tab_abl2.tex", "\n".join(lines))

    # --- compute cost
    cost = M[M.policy.isin(["zs", "static", "dtr_m8", "active_even1_m8"])].groupby("policy")[["ms_per_email_uncached", "peak_gpu_gb"]].mean().round(1)
    cost.to_csv(os.path.join(a.out, "cost.csv"))

    # --- weekly figure data
    PP = pd.DataFrame([dict(policy=pol, seed=s, period=w, roca=one_minus_roca_pct(g.y, g.score), f1=100 * metrics(g.y, g.pred)["f1"])
                       for (pol, s, w), g in P.groupby(["policy", "seed", "period"])])
    Wk = PP.groupby(["policy", "period"])[["roca", "f1"]].mean().reset_index()
    def dat(pols, col):
        t = Wk[Wk.policy.isin(pols)].pivot(index="period", columns="policy", values=col)[pols].round(3)
        return "period " + " ".join(pols) + "\n" + "\n".join(f"{i} " + " ".join(str(v) for v in r) for i, r in zip(t.index, t.values))
    W("dat_drift_clf.txt", dat(["clf_static_b50", "clf_static_b500", "clf_full"], "roca"))
    W("dat_drift_llm.txt", dat(["static", "recent_even1_m8", "dtr_m8"], "roca"))
    W("dat_f1_llm.txt", dat(["zs", "static", "recent_even1_m8", "recent_R1", "dtr_m8", "dtr_m6"], "f1"))
    W("dat_f1_clf.txt", dat(["clf_static_b6", "clf_static_b50", "clf_static_b500", "clf_recent_b50", "clf_recent_b500", "clf_full"], "f1"))
    w = Wk.pivot(index="period", columns="policy", values="roca"); w.round(3).to_csv(os.path.join(a.out, "weekly_roca.csv"))

    # --- cost / Pareto figure data
    S = CI.join(meta.labels_used).dropna(subset=["labels_used"]); S = S.sort_values("labels_used")
    best = np.inf; S["pareto"] = 0
    for i, r in S.iterrows():
        if r.roca < best: best = r.roca; S.loc[i, "pareto"] = 1
    S.to_csv(os.path.join(a.out, "fig_cost.csv"))
    fam = lambda p: "classical" if p.startswith(("clf_", "inc_")) else "llm"
    llm = [p for p in S.index if fam(p) == "llm" and p in LLM_MAIN]; clf = [p for p in S.index if fam(p) == "classical" and p in CLF_MAIN]
    def pts(ps): return "\n".join(f"{max(S.loc[p,'labels_used'],1):g} {S.loc[p,'roca']:.3f} {S.loc[p,'roca']-S.loc[p,'roca_lo']:.3f} {S.loc[p,'roca_hi']-S.loc[p,'roca']:.3f}" for p in ps)
    front = S[S.pareto == 1]
    W("dat_cost_llm.txt", pts(llm)); W("dat_cost_clf.txt", pts(clf))
    W("dat_cost_front.txt", "\n".join(f"{max(r.labels_used,1):g} {r.roca:.3f}" for _, r in front.iterrows()))
    W("cost_front_labels.txt", "\n".join(f"{p} {max(r.labels_used,1):g} {r.roca:.3f}" for p, r in front.iterrows()))

    # --- labels used vs inspected (bar chart data)
    lb = ["static", "fixed_w1", "dtr_m8", "recent_even1_m8", "randtrig_m8", "active_even1_m8", "dtr_m6", "recent_R3", "recent_R1"]
    W("dat_labels.txt", "x used inspected\n" + "\n".join(f"{i} {meta.loc[p,'labels_used']:.1f} {meta.loc[p,'labels_inspected']:.1f}" for i, p in enumerate(lb)))
    W("labels_xticks.txt", ",".join(nm(p) for p in lb))

    # --- detector table + bar chart data
    d = pd.read_csv(os.path.join(a.synthetic, "detector_runs.csv"))
    out = []
    for det, g in d.groupby("detector"):
        dr = g[g.scenario != "none"]
        rate = {sc: dr[dr.scenario == sc].detected.sum() / max(1, (dr[dr.scenario == sc].detected + dr[dr.scenario == sc].missed).sum()) for sc in ("sudden", "gradual", "recurring")}
        out.append(dict(detector=det, sudden=rate["sudden"], gradual=rate["gradual"], recurring=rate["recurring"],
                        fa_none=g[g.scenario == "none"].false_alarms.mean(), fa_drift=dr.false_alarms.mean(), delay=dr.mean_delay.mean()))
    Dt = pd.DataFrame(out).set_index("detector"); Dt.round(3).to_csv(os.path.join(a.out, "detectors.csv"))
    DN = {"char_centroid_m6": "Char centroid, $m{=}6$ (ours)", "char_centroid_m8": "Char centroid, $m{=}8$ (ours)",
          "char_centroid_m2": "Char centroid, $m{=}2$", "word_tfidf_centroid_m8": "Word TF-IDF centroid, $m{=}8$",
          "embedding_mmd": "Embedding MMD", "ks_classifier_score": "KS on model scores", "page_hinkley_score": "Page-Hinkley on scores",
          "ddm_errors_LABELS": "DDM (needs labels)"}
    lines = [f"{DN[k]} & {100*Dt.loc[k,'sudden']:.0f} & {100*Dt.loc[k,'gradual']:.0f} & {100*Dt.loc[k,'recurring']:.0f} & {Dt.loc[k,'fa_none']:.1f} & "
             + ("--" if pd.isna(Dt.loc[k, 'delay']) else f"{Dt.loc[k,'delay']:.2f}") + BS + BS for k in DN if k in Dt.index]
    W("tab_det.tex", "\n".join(lines))
    bd = ["char_centroid_m6", "char_centroid_m8", "word_tfidf_centroid_m8", "embedding_mmd", "ks_classifier_score", "page_hinkley_score", "ddm_errors_LABELS"]
    W("dat_det.txt", "x det fa\n" + "\n".join(f"{i} {100*(Dt.loc[k,['sudden','gradual','recurring']].mean()):.1f} {Dt.loc[k,'fa_none']:.2f}" for i, k in enumerate(bd)))
    print(mean[["roca", "f1", "hm", "sm"]].round(2).to_string()); print(cost); print(Dt.round(2).to_string())


if __name__ == "__main__":
    main()
