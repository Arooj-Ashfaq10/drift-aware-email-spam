"""Step 3 (CPU): classical baseline (TF-IDF + logistic regression) under the SAME stream, eval sample and label budgets."""
import argparse, os
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.email_data import build_email_stream
from src.strategies import window, history


def balanced_sample(pool, b, seed):
    rng = np.random.default_rng(seed)
    sp = pool[pool.label == 1]; hm = pool[pool.label == 0]
    sp = sp.iloc[rng.permutation(len(sp))[: b // 2]]; hm = hm.iloc[rng.permutation(len(hm))[: b - b // 2]]
    return pd.concat([sp, hm])


def fit_score(train, texts):
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=50000)
    lr = LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(vec.fit_transform(train.msg_text), train.label)
    return lr.decision_function(vec.transform(texts))


def run(stream, name, kind, b, seed, cap=20000):
    P = int(stream.period.max()) + 1; train = None; refreshes = 0; rows = []
    for t in range(P):
        refreshed = False
        if kind == "full":
            h = window(stream, 0) if t == 0 else history(stream, t)
            if len(h) > cap: h = h.sample(cap, random_state=seed)
            train, refreshed = h, t >= 1
        elif train is None:
            train = balanced_sample(window(stream, 0), b, seed)
        elif kind == "recent" and t >= 2:
            train = balanced_sample(window(stream, t - 1), b, seed * 1000 + t); refreshed = True; refreshes += 1
        ev = stream[(stream.period == t) & stream.evalset]
        sc = fit_score(train, ev.msg_text.tolist()); pred = (sc > 0).astype(int)
        for mid, y, p_, s_ in zip(ev.id, ev.label, pred, sc):
            rows.append(dict(policy=name, seed=seed, period=t, id=int(mid), y=int(y), pred=int(p_), score=float(s_), refreshed=refreshed))
    labels = len(train) if kind == "full" else b * (1 + refreshes)
    return pd.DataFrame(rows), dict(policy=name, seed=seed, refreshes=refreshes, labels_used=labels, schedule="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--periods", type=int, default=12); ap.add_argument("--eval_per_class", type=int, default=50)
    ap.add_argument("--ex_chars", type=int, default=300); ap.add_argument("--msg_chars", type=int, default=500)
    ap.add_argument("--budgets", type=int, nargs="+", default=[6, 50, 500]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    stream = build_email_stream(a.parsed, a.periods, a.eval_per_class, ex_chars=a.ex_chars, msg_chars=a.msg_chars)
    preds, metas = [], []
    for b in a.budgets:
        for kind in ("static", "recent"):
            for s in a.seeds:
                p, m = run(stream, f"clf_{kind}_b{b}", kind, b, s); preds.append(p); metas.append(m)
        print(f"budget {b} done", flush=True)
    for s in a.seeds[:1]:
        p, m = run(stream, "clf_full", "full", 0, s); preds.append(p); metas.append(m)
    pd.concat(preds).to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    pd.DataFrame(metas).to_csv(os.path.join(a.out, "run_meta.csv"), index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
