"""Demonstration selection, labelled windows, refresh schedules and the label-free content-drift monitor (v5).

Selection (comments 23, 24, 26, 35): an oracle inspects candidate emails in a strategy-specific ORDER and reveals each label;
inspection stops once k/2 ham and k/2 spam are found. Every inspected email costs one label (inspected cost), the k kept
ones are the used cost. Near-duplicates (char n-gram cosine > dup_thr to an already chosen demonstration) are skipped.
Orders: random | newest (newest first) | similarity (closest to the newest week's unlabelled centroid first)
        | diversity (farthest-first traversal) | uncertainty (smallest |LLM score| under the current prompt first)."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 16, alternate_sign=False, norm="l2")


def vec(texts):
    return _HV.transform(list(texts))


def window(stream, p):
    """Labelled emails of period p (period 0: only the labelled pool part)."""
    w = stream[stream.period == p]
    return w[w.split == "pool"] if p == 0 else w


def history(stream, t):
    h = stream[stream.period < t]
    return h[~((h.period == 0) & (h.split == "eval"))]


def _order(pool, how, rng, cand_cap=400, ref_centroid=None, scorer=None):
    """Return candidate row positions (into pool) in inspection order."""
    n = len(pool)
    if how == "random" or n == 0:
        return rng.permutation(n)
    if how == "newest":
        return np.argsort(-pool.recv.values.astype("datetime64[s]").astype(np.int64), kind="stable")
    sub = rng.permutation(n)[: min(cand_cap, n)]            # unlabelled candidate sample
    X = vec(pool.ex_text.values[sub])
    if how == "similarity":
        c = ref_centroid if ref_centroid is not None else np.asarray(X.mean(axis=0)).ravel()
        return sub[np.argsort(-(X @ c))]
    if how == "diversity":
        first = int(np.argmax(X @ np.asarray(X.mean(axis=0)).ravel()))
        order, d = [first], 1.0 - (X @ X[first].T).toarray().ravel()
        for _ in range(len(sub) - 1):
            j = int(np.argmax(d)); order.append(j); d = np.minimum(d, 1.0 - (X @ X[j].T).toarray().ravel())
        return sub[np.array(order)]
    if how == "uncertainty":
        s = scorer(pool.msg_text.values[sub].tolist())
        return sub[np.argsort(np.abs(s))]
    raise ValueError(how)


def select(pool, k, seed, how="random", dup_thr=0.9, keep=None, **kw):
    """Inspect candidates in order until k/2 per class are found. keep: list of (text, label) old demos to retain.
    Returns (examples, inspected, used)."""
    rng = np.random.default_rng(seed)
    keep = list(keep or [])
    need = {1: k // 2 - sum(1 for _, y in keep if y == 1), 0: (k - k // 2) - sum(1 for _, y in keep if y == 0)}
    chosen, chosen_vec, inspected = list(keep), [vec([t]) for t, _ in keep], 0
    for i in _order(pool, how, rng, **kw):
        if need[0] <= 0 and need[1] <= 0:
            break
        r = pool.iloc[int(i)]; inspected += 1; y = int(r.label)
        if need[y] <= 0:
            continue
        v = vec([r.ex_text])
        if dup_thr and any(float((v @ c.T).toarray()[0, 0]) > dup_thr for c in chosen_vec):
            continue
        chosen.append((r.ex_text, y)); chosen_vec.append(v); need[y] -= 1
    order = rng.permutation(len(chosen))
    return [chosen[j] for j in order], inspected, len(chosen) - len(keep)


def keep_closest(examples, ref_centroid, n_keep):
    """Partial replacement: keep the n_keep old demonstrations closest to the newest week's centroid (comment 22)."""
    if n_keep <= 0 or not examples:
        return []
    sims = (vec([t for t, _ in examples]) @ ref_centroid).ravel()
    return [examples[i] for i in np.argsort(-sims)[:n_keep]]


def centroid(texts):
    c = np.asarray(vec(texts).mean(axis=0)).ravel()
    return c / (np.linalg.norm(c) + 1e-12)


def period_stats(stream, cap=3000, chars=500, seed=0, n_splits=20):
    """Per period: unit centroid of the UNLABELLED emails and the sampling-noise floor (mean split-half distance)."""
    rng = np.random.default_rng(seed); cents, noise = {}, {}
    for p in sorted(stream.period.unique()):
        sub = stream[stream.period == p]
        if len(sub) > cap:
            sub = sub.iloc[rng.choice(len(sub), cap, replace=False)]
        X = vec((sub.subject + " " + sub.body.str[:chars]).tolist())
        c = np.asarray(X.mean(axis=0)).ravel(); cents[int(p)] = c / (np.linalg.norm(c) + 1e-12)
        d = []
        for _ in range(n_splits):
            perm = rng.permutation(X.shape[0]); h = X.shape[0] // 2
            a = np.asarray(X[perm[:h]].mean(axis=0)).ravel(); b = np.asarray(X[perm[h:]].mean(axis=0)).ravel()
            d.append(1.0 - float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        noise[int(p)] = float(np.mean(d))
    return cents, noise


def dtr_schedule(cents, noise, mult, cooldown=1, persistence=1, reset=True):
    """Label-free, causal Drift-Triggered Refresh schedule. After each period p = 1..T-2: d = 1 - cos(c[ref], c[p]);
    alarm if d > mult * noise[ref] for `persistence` consecutive periods and at least `cooldown` periods since the last
    refresh. A refresh before period p+1 uses examples from period p. reset=False keeps the Week-0 reference (ablation)."""
    T = len(cents); ref, refresh, traj, streak, last = 0, [], [], 0, -10 ** 9
    for p in range(1, T - 1):
        d = 1.0 - float(cents[ref] @ cents[p]); theta = mult * noise[ref]; traj.append((p, d, theta))
        streak = streak + 1 if d > theta else 0
        if streak >= persistence and (p + 1) - last >= cooldown:
            refresh.append(p + 1); last = p + 1; streak = 0
            if reset:
                ref = p
    return refresh, traj


def periodic_schedule(T, R):
    return [t for t in range(2, T) if (t - 1) % R == 0]


def even_schedule(T, n, start=2):
    """Equal-budget control for DTR: n refreshes spread evenly (centred) over periods start..T-1."""
    if n <= 0:
        return []
    span = T - start
    return sorted(set(min(T - 1, start + int(np.floor((i + 0.5) * span / n))) for i in range(n)))


def random_schedule(T, n, seed, start=2):
    """Ablation: same number of refreshes as DTR at random times (comment 51)."""
    rng = np.random.default_rng(10_000 + seed)
    return sorted(rng.choice(np.arange(start, T), size=min(n, T - start), replace=False).tolist())
