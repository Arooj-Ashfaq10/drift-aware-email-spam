"""CPU: detector evaluation on controlled drift streams + drift report for the real Week 0 -> 1 boundary.

Comments 17 (drift evidence), 18/40 (gradual, recurring, no-drift streams with known location and size),
19/21 (alternative representations and detectors, label-free vs label-based), 20 (noise floor vs number of splits),
41 (detector judged on its own, independent of classification).

Controlled streams: 12 periods x n emails. Each email comes from source B (later weeks 6-11) with probability pi_t,
otherwise from source A (Week 0). Scenarios (mag in {0.25, 0.5, 1.0}):
  none: pi=0 | sudden: pi=mag from t=6 | gradual: pi ramps 0->mag over t=3..9 | recurring: pi=mag at t in {4,5,8,9}
True change points: sudden {6}; gradual {3} (onset); recurring {4,6,8,10}. An alarm within `tol` periods after a change
point detects it (delay = alarm - change); other alarms are false alarms; changes without an alarm are missed.

usage: python synthetic_drift.py --parsed trec07p_parsed.csv.gz --out results/synthetic [--seeds 0 1 2 3 4]"""
import argparse, os, re
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from src.email_data import build_email_stream
from src.strategies import vec, period_stats

T = 12; CHANGES = {"none": [], "sudden": [6], "gradual": [3], "recurring": [4, 6, 8, 10]}


def pis(scn, mag):
    t = np.arange(T)
    if scn == "none": return np.zeros(T)
    if scn == "sudden": return np.where(t >= 6, mag, 0.0)
    if scn == "gradual": return np.clip((t - 2) / 7, 0, 1) * mag
    if scn == "recurring": return np.where(np.isin(t, [4, 5, 8, 9]), mag, 0.0)


def make_stream(A, B, scn, mag, n, rng):
    parts = []
    for t, p in enumerate(pis(scn, mag)):
        nb = rng.binomial(n, p); na = n - nb
        a = A.iloc[rng.choice(len(A), na, replace=False)]; b = B.iloc[rng.choice(len(B), nb, replace=False)]
        parts.append(pd.concat([a, b]).assign(period=t))
    return pd.concat(parts, ignore_index=True)


# ---------------- detectors: each returns the list of alarm periods (1..T-1) ----------------
def centroid_detector(X_by_t, mult, n_splits=20, seed=0):
    rng = np.random.default_rng(seed); cents, noise = [], []
    for X in X_by_t:
        c = np.asarray(X.mean(axis=0)).ravel(); cents.append(c / (np.linalg.norm(c) + 1e-12)); d = []
        for _ in range(n_splits):
            pr = rng.permutation(X.shape[0]); h = X.shape[0] // 2
            a = np.asarray(X[pr[:h]].mean(axis=0)).ravel(); b = np.asarray(X[pr[h:]].mean(axis=0)).ravel()
            d.append(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        noise.append(np.mean(d))
    ref, al = 0, []
    for t in range(1, T):
        if 1 - cents[ref] @ cents[t] > mult * noise[ref]:
            al.append(t); ref = t
    return al


def ks_detector(S_by_t, alpha=0.01):
    ref, al = 0, []
    for t in range(1, T):
        if ks_2samp(S_by_t[ref], S_by_t[t]).pvalue < alpha:
            al.append(t); ref = t
    return al


def mmd_detector(E_by_t, rng, n_perm=200, alpha=0.01, cap=300):
    def mmd(X, Y):
        Z = np.vstack([X, Y]); K = Z @ Z.T; n = len(X)
        return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean(), K, n
    ref, al = 0, []
    for t in range(1, T):
        X = E_by_t[ref][:cap]; Y = E_by_t[t][:cap]; stat, K, n = mmd(X, Y); cnt = 0
        for _ in range(n_perm):
            pr = rng.permutation(len(K)); Kp = K[np.ix_(pr, pr)]
            cnt += (Kp[:n, :n].mean() + Kp[n:, n:].mean() - 2 * Kp[:n, n:].mean()) >= stat
        if (cnt + 1) / (n_perm + 1) < alpha:
            al.append(t); ref = t
    return al


def page_hinkley(x_by_t, delta=0.005, lam=None):
    """Two-sided Page-Hinkley on the email-level score stream (label-free); alarm reported at the period it occurs."""
    xs = np.concatenate(x_by_t); per = np.concatenate([[t] * len(x) for t, x in enumerate(x_by_t)])
    lam = lam if lam is not None else 50 * np.std(x_by_t[0])
    mean, mt_u, mt_l, Mu, Ml, n, al = 0.0, 0.0, 0.0, 0.0, 0.0, 0, []
    for x, t in zip(xs, per):
        n += 1; mean += (x - mean) / n
        mt_u += x - mean - delta; Mu = min(Mu, mt_u); mt_l += mean - x - delta; Ml = min(Ml, mt_l)
        if t >= 1 and (mt_u - Mu > lam or mt_l - Ml > lam):
            al.append(int(t)); mean, mt_u, mt_l, Mu, Ml, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    return sorted(set(al))


def ddm(err_by_t, warn=2.0, drift=3.0):
    """DDM on the classifier's error stream (LABEL-BASED reference detector)."""
    al, n, p, pmin, smin = [], 0, 0.0, np.inf, np.inf
    for t, errs in enumerate(err_by_t):
        for e in errs:
            n += 1; p += (e - p) / n; s = np.sqrt(p * (1 - p) / n)
            if n > 30 and p + s < pmin + smin: pmin, smin = p, s
            if n > 30 and t >= 1 and p + s > pmin + drift * smin:
                al.append(t); n, p, pmin, smin = 0, 0.0, np.inf, np.inf
    return sorted(set(al))


def score_alarms(al, changes, tol=2):
    hit, delays, used = 0, [], set()
    for c in changes:
        cand = [a for a in al if c <= a <= c + tol and a not in used]
        if cand: hit += 1; delays.append(cand[0] - c); used.add(cand[0])
    return dict(detected=hit, missed=len(changes) - hit, false_alarms=len([a for a in al if a not in used]),
                mean_delay=float(np.mean(delays)) if delays else np.nan)


def drift_report(s, out):
    """Top changed words and example subjects across the real Week 0 -> 1 boundary (comment 17). Emails, URLs and long
    digit strings are masked."""
    mask = lambda x: re.sub(r"\d{3,}", "<num>", re.sub(r"https?://\S+|www\.\S+", "<url>", re.sub(r"\S+@\S+", "<email>", str(x))))
    w0, w1 = s[s.period == 0], s[s.period == 1]
    cv = CountVectorizer(ngram_range=(1, 2), min_df=20, binary=True, stop_words="english")
    X = cv.fit_transform(pd.concat([w0, w1]).msg_text.map(mask)); n0 = len(w0)
    f0 = np.asarray(X[:n0].mean(axis=0)).ravel(); f1 = np.asarray(X[n0:].mean(axis=0)).ravel(); terms = np.array(cv.get_feature_names_out())
    d = f1 - f0; up, down = np.argsort(-d)[:20], np.argsort(d)[:20]
    rows = [dict(term=terms[i], df_week0=f0[i], df_week1=f1[i], change=d[i]) for i in np.r_[down, up]]
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "drift_terms_week0_week1.csv"), index=False)
    with open(os.path.join(out, "drift_examples_week0_week1.txt"), "w", encoding="utf8") as f:
        for name, w in (("Week 0", w0), ("Week 1", w1)):
            for y, lab in ((1, "spam"), (0, "ham")):
                f.write(f"== {name} {lab} (spam share {w.label.mean():.3f}, n={len(w)}) ==\n")
                for subj in w[w.label == y].sample(8, random_state=0).subject:
                    f.write("  " + mask(subj)[:90] + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_period", type=int, default=1000); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    s = build_email_stream(a.parsed, T, 50)
    s["mon_text"] = s.subject + " " + s.body.str[:500]

    # (20) noise-floor stability vs number of half-splits
    rows = []
    for ns in (2, 5, 10, 20, 50, 100):
        _, nz = period_stats(s, n_splits=ns); rows.append(dict(n_splits=ns, **{f"w{p}": round(v, 5) for p, v in nz.items()}))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "noise_floor_vs_splits.csv"), index=False); print(pd.DataFrame(rows).to_string())
    drift_report(s, a.out)

    A, B = s[s.period == 0], s[s.period >= 6]
    try:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(a.embed_model)
    except Exception as e:
        print("sentence-transformers not available, MMD detector skipped:", e); emb = None
    res = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        for scn in CHANGES:
            for mag in ([1.0] if scn == "none" else [0.25, 0.5, 1.0]):
                st = make_stream(A, B, scn, mag, a.n_per_period, rng)
                by_t = [st[st.period == t] for t in range(T)]
                Xc = [vec(g.mon_text) for g in by_t]
                tf = TfidfVectorizer(sublinear_tf=True, min_df=2).fit(by_t[0].mon_text); Xw = [tf.transform(g.mon_text) for g in by_t]
                lr = LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(tf.transform(by_t[0].msg_text), by_t[0].label)
                S = [lr.decision_function(tf.transform(g.msg_text)) for g in by_t]
                E = [(lr.predict(tf.transform(g.msg_text)) != g.label.values).astype(int) for g in by_t]
                dets = {f"char_centroid_m{m}": centroid_detector(Xc, m, seed=seed) for m in (2, 4, 6, 8, 16)}
                dets.update({f"word_tfidf_centroid_m{m}": centroid_detector(Xw, m, seed=seed) for m in (2, 4, 6, 8, 16)})
                dets["ks_classifier_score"] = ks_detector(S); dets["page_hinkley_score"] = page_hinkley(S); dets["ddm_errors_LABELS"] = ddm(E)
                if emb is not None:
                    Em = [emb.encode(g.mon_text.str[:512].tolist()[:300], normalize_embeddings=True, batch_size=64) for g in by_t]
                    dets["embedding_mmd"] = mmd_detector(Em, rng)
                for name, al in dets.items():
                    res.append(dict(seed=seed, scenario=scn, mag=mag, detector=name, alarms=" ".join(map(str, al)), **score_alarms(al, CHANGES[scn])))
        print("seed done", seed, flush=True)
    R = pd.DataFrame(res); R.to_csv(os.path.join(a.out, "detector_runs.csv"), index=False)
    Sm = R.groupby(["detector", "scenario", "mag"])[["detected", "missed", "false_alarms", "mean_delay"]].mean().round(2)
    Sm.to_csv(os.path.join(a.out, "detector_summary.csv")); print(Sm.to_string())


if __name__ == "__main__":
    main()
