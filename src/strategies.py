"""Example selection, labelled windows, and the label-free content-drift monitor (DTR) for email streams."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 16, alternate_sign=False, norm="l2")


def pick_examples(pool, k, seed=0):
    """Class-balanced random draw: k/2 spam + k/2 ham from `pool` (uses the ex_text column)."""
    rng = np.random.default_rng(seed)
    sp = pool[pool.label == 1]; hm = pool[pool.label == 0]
    sp = sp.iloc[rng.permutation(len(sp))[: k // 2]]; hm = hm.iloc[rng.permutation(len(hm))[: k - k // 2]]
    ex = pd.concat([sp, hm]); ex = ex.iloc[rng.permutation(len(ex))]
    return list(zip(ex.ex_text.tolist(), ex.label.tolist()))


def window(stream, p):
    """Labelled emails of period p (period 0: only the labelled pool part)."""
    w = stream[stream.period == p]
    return w[w.split == "pool"] if p == 0 else w


def history(stream, t):
    h = stream[stream.period < t]
    return h[~((h.period == 0) & (h.split == "eval"))]


def _centroid(texts):
    c = np.asarray(_HV.transform(texts).mean(axis=0)).ravel()
    return c / (np.linalg.norm(c) + 1e-12)


def period_stats(stream, cap=3000, chars=500, seed=0, n_splits=20):
    """Per period: unit content centroid of the UNLABELLED emails and the sampling-noise floor, i.e. the mean content distance
    between two random halves of the same period (what pure sampling noise produces)."""
    rng = np.random.default_rng(seed); cents, noise = {}, {}
    for p in sorted(stream.period.unique()):
        sub = stream[stream.period == p]
        if len(sub) > cap:
            sub = sub.iloc[rng.choice(len(sub), cap, replace=False)]
        X = _HV.transform((sub.subject + " " + sub.body.str[:chars]).tolist())
        c = np.asarray(X.mean(axis=0)).ravel(); cents[int(p)] = c / (np.linalg.norm(c) + 1e-12)
        d = []
        for _ in range(n_splits):
            perm = rng.permutation(X.shape[0]); h = X.shape[0] // 2
            a = np.asarray(X[perm[:h]].mean(axis=0)).ravel(); b = np.asarray(X[perm[h:]].mean(axis=0)).ravel()
            d.append(1.0 - float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        noise[int(p)] = float(np.mean(d))
    return cents, noise


def period_centroids(stream, cap=3000, chars=500, seed=0):
    return period_stats(stream, cap, chars, seed, n_splits=2)[0]


def dtr_schedule(cents, noise, mult):
    """Label-free, causal Drift-Triggered Refresh schedule on the UNLABELLED incoming stream (no warm-up needed).
    After each period p = 1..T-2: d = 1 - cos(centroid(ref), centroid(p)). Refresh the prompt examples for period p+1 (examples drawn
    from period p) if d > theta, with theta = mult * noise[ref], i.e. `mult` times the content distance that pure sampling noise
    produces inside the reference period. After a refresh the reference becomes period p. The first reference is period 0."""
    T = len(cents); ref, refresh, traj = 0, [], []
    for p in range(1, T - 1):
        d = 1.0 - float(cents[ref] @ cents[p]); theta = mult * noise[ref]; traj.append((p, d, theta))
        if d > theta:
            refresh.append(p + 1); ref = p
    return refresh, traj


def periodic_schedule(T, R):
    return [t for t in range(2, T) if (t - 1) % R == 0]


def even_schedule(T, n, start=2):
    """Equal-budget control for DTR: n refreshes spread evenly (centred) over periods start..T-1."""
    if n <= 0:
        return []
    span = T - start
    return sorted(set(min(T - 1, start + int(np.floor((i + 0.5) * span / n))) for i in range(n)))
