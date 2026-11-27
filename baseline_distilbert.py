"""GPU (training): lightweight transformer baseline (comment 34). DistilBERT fine-tuned on the same label budgets as the
classical baselines: static (b Week-0 labels, trained once) and recent (re-trained on the b newest-week labels before each
week from Week 2). Writes predictions.csv / run_meta.csv in the same format as the other runners, so analyze*.py work.

usage: python baseline_distilbert.py --parsed trec07p_parsed.csv.gz --out results/distilbert --budgets 50 500 --seeds 0 1 2"""
import argparse, os, time
import numpy as np, pandas as pd
from src.email_data import build_email_stream
from src.strategies import window
from run_classical import balanced_sample


class Bert:
    def __init__(self, seed, name="distilbert-base-uncased", epochs=4, lr=5e-5, bs=16, max_len=256):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.manual_seed(seed); self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(self.dev)
        self.epochs, self.lr, self.bs, self.max_len, self.rng = epochs, lr, bs, max_len, np.random.default_rng(seed)

    def fit(self, texts, labels):
        torch = self.torch; opt = torch.optim.AdamW(self.m.parameters(), lr=self.lr); self.m.train(); texts, labels = list(texts), np.asarray(labels)
        for _ in range(self.epochs):
            order = self.rng.permutation(len(texts))
            for i in range(0, len(texts), self.bs):
                idx = order[i:i + self.bs]
                enc = self.tok([texts[j] for j in idx], truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.dev)
                loss = self.m(**enc, labels=torch.tensor(labels[idx], device=self.dev)).loss; loss.backward(); opt.step(); opt.zero_grad()
        return self

    def score(self, texts, bs=64):
        torch = self.torch; self.m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self.tok(list(texts[i:i + bs]), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.dev)
                lg = self.m(**enc).logits.float(); out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
        return np.concatenate(out)


def run(stream, kind, b, seed):
    P = int(stream.period.max()) + 1; rows = []; refreshes = 0
    tr = balanced_sample(window(stream, 0), b, seed); tr = tr.sample(frac=1, random_state=seed)
    model = Bert(seed).fit(tr.msg_text, tr.label)
    for t in range(P):
        refreshed = False
        if kind == "recent" and t >= 2:
            tr = balanced_sample(window(stream, t - 1), b, seed * 1000 + t).sample(frac=1, random_state=seed)
            model = Bert(seed).fit(tr.msg_text, tr.label); refreshed = True; refreshes += 1
        ev = stream[(stream.period == t) & stream.evalset]; sc = model.score(ev.msg_text.tolist()); pred = (sc > 0).astype(int)
        rows += [dict(policy=f"bert_{kind}_b{b}", seed=seed, period=t, id=int(i), y=int(y), pred=int(p), score=float(s), refreshed=refreshed)
                 for i, y, p, s in zip(ev.id, ev.label, pred, sc)]
    return pd.DataFrame(rows), dict(policy=f"bert_{kind}_b{b}", seed=seed, refreshes=refreshes, labels_used=b * (1 + refreshes), schedule="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[50, 500]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--kinds", nargs="+", default=["static", "recent"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    stream = build_email_stream(a.parsed, 12, 50)
    pp, mp = os.path.join(a.out, "predictions.csv"), os.path.join(a.out, "run_meta.csv")
    done = set(zip(*pd.read_csv(mp)[["policy", "seed"]].values.T)) if os.path.exists(mp) else set()
    for b in a.budgets:
        for kind in a.kinds:
            for s in a.seeds:
                if (f"bert_{kind}_b{b}", s) in done: continue
                t0 = time.time(); p, m = run(stream, kind, b, s); m["seconds"] = round(time.time() - t0)
                p.to_csv(pp, mode="a", header=not os.path.exists(pp), index=False); pd.DataFrame([m]).to_csv(mp, mode="a", header=not os.path.exists(mp), index=False)
                print(m, flush=True)


if __name__ == "__main__":
    main()
