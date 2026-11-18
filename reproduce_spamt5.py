"""Base-paper reproduction: Spam-T5 (Labonne & Moran, arXiv:2304.01238) - Flan-T5-base fine-tuned for email spam, few-shot and full.
Re-implemented from the published description (text-to-text; batch 8, lr 5e-5, 5 epochs; random 80/20 split; F1/precision/recall).
Deviations (documented in the paper): dataset versions from the Zenodo curated bundle, input format, max length, test-set cap,
'k' = training emails PER CLASS, fp32 training. --dry uses a TF-IDF classifier instead of T5 to test the pipeline on CPU."""
import argparse, os, re, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df[["subject", "body"]] = df[["subject", "body"]].fillna("").astype(str)
    df["text"] = ("Subject: " + df.subject + " | " + df.body).str.replace(r"\s+", " ", regex=True)
    return df[["text", "label"]].drop_duplicates(subset="text").reset_index(drop=True)


class T5Clf:
    def __init__(self, epochs=5, bs=8, lr=5e-5, max_len=256, seed=0, name="google/flan-t5-base"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.torch, self.epochs, self.bs, self.lr, self.max_len, self.seed = torch, epochs, bs, lr, max_len, seed
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name); self.name = name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.dev)      # fp32 (T5 is unstable in fp16)
        for ws, wh in (("spam", "ham"), ("yes", "no")):          # first label pair whose first sentencepiece tokens differ
            si, hi = self.tok(ws, add_special_tokens=False).input_ids[0], self.tok(wh, add_special_tokens=False).input_ids[0]
            if si != hi:
                self.words, self.spam_id, self.ham_id = (ws, wh), si, hi; break
        else:
            raise RuntimeError("no usable label words")

    def _enc(self, texts):
        return self.tok(["classify email: " + t for t in texts], max_length=self.max_len, truncation=True, padding=True, return_tensors="pt").to(self.dev)

    def fit(self, texts, labels):
        torch = self.torch; torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr); self.model.train()
        texts, labels = list(texts), list(labels)
        for ep in range(self.epochs):
            order = rng.permutation(len(texts))
            for i in range(0, len(order), self.bs):
                idx = order[i:i + self.bs]; enc = self._enc([texts[j] for j in idx])
                tgt = self.tok([self.words[0] if labels[j] == 1 else self.words[1] for j in idx], padding=True, return_tensors="pt").input_ids.to(self.dev)
                tgt[tgt == self.tok.pad_token_id] = -100
                loss = self.model(**enc, labels=tgt).loss; loss.backward(); opt.step(); opt.zero_grad()
        return self

    def score(self, texts, bs=32):
        torch = self.torch; self.model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self._enc(list(texts[i:i + bs]))
                dec = torch.full((enc.input_ids.shape[0], 1), self.model.config.decoder_start_token_id, device=self.dev)
                lg = self.model(**enc, decoder_input_ids=dec).logits[:, 0, :]
                out.append((lg[:, self.spam_id] - lg[:, self.ham_id]).float().cpu().numpy())
        return np.concatenate(out)


class DryClf:
    def __init__(self, **kw): pass
    def fit(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.v = TfidfVectorizer(sublinear_tf=True); self.m = LogisticRegression(max_iter=1000).fit(self.v.fit_transform(texts), labels); return self
    def score(self, texts): return self.m.decision_function(self.v.transform(texts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True, help="name=path pairs, e.g. ling=/content/Ling.csv")
    ap.add_argument("--ks", nargs="+", default=["8", "32", "128", "full"]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_test", type=int, default=2000); ap.add_argument("--max_full", type=int, default=6000, help="cap on training emails for 'full'")
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, "spamt5_reproduction.csv")
    done = set(); rows = []
    if os.path.exists(path):
        r0 = pd.read_csv(path); rows = r0.to_dict("records"); done = set(zip(r0.dataset, r0.k.astype(str), r0.seed))
    for item in a.csvs:
        name, p = item.split("=", 1); df = load_csv(p)
        tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
        if len(te) > a.max_test: te = te.sample(a.max_test, random_state=0)
        print(f"{name}: {len(df)} emails (spam {df.label.mean():.1%}) | train pool {len(tr)} | test {len(te)}", flush=True)
        for k in a.ks:
            for s in a.seeds:
                if (name, str(k), s) in done: print("skip", name, k, s); continue
                if k == "full":
                    sub = tr if len(tr) <= a.max_full else tr.sample(a.max_full, random_state=s)
                else:
                    sub = pd.concat([g.sample(min(int(k), len(g)), random_state=s) for _, g in tr.groupby("label")])
                t0 = time.time(); clf = (DryClf() if a.dry else T5Clf(epochs=a.epochs, max_len=a.max_len, seed=s)).fit(sub.text.tolist(), sub.label.tolist())
                pred = (clf.score(te.text.tolist()) > 0).astype(int)
                rows.append(dict(dataset=name, k=k, seed=s, n_train=len(sub), n_test=len(te), f1=f1_score(te.label, pred), precision=precision_score(te.label, pred, zero_division=0),
                                 recall=recall_score(te.label, pred), seconds=round(time.time() - t0)))
                pd.DataFrame(rows).to_csv(path, index=False); print(rows[-1], flush=True)
                if not a.dry:
                    import torch; del clf; torch.cuda.empty_cache()
    R = pd.DataFrame(rows); g = R.groupby(["dataset", "k"])[["f1", "precision", "recall"]].agg(["mean", "std"]).round(4); print(g.to_string()); g.to_csv(os.path.join(a.out, "spamt5_summary.csv"))


if __name__ == "__main__":
    main()
