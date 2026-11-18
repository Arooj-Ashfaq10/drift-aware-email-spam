"""Builds a tiny TREC-like corpus (data/, full/index) with varied MIME structure, forged Date headers and drifting spam."""
import base64, datetime as dt, os, sys
import numpy as np

root = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
os.makedirs(f"{root}/data", exist_ok=True); os.makedirs(f"{root}/full", exist_ok=True)
rng = np.random.default_rng(0)
fam = [["viagra", "pills", "cheap", "pharmacy", "order"], ["lottery", "winner", "prize", "claim", "bank"], ["stock", "alert", "invest", "penny", "profit"]]
ham = "meeting schedule project report lunch tomorrow thanks review draft agenda call team update minutes".split()
t0 = dt.datetime(2007, 4, 8, 13, 7, 21); span = 89 * 86400
lines = []
for i in range(n):
    frac = i / n; t = t0 + dt.timedelta(seconds=frac * span + rng.integers(0, 60))
    spam = rng.random() < 0.666
    if spam:
        f = min(int(frac * 3 + rng.normal(0, 0.3)), 2); f = max(f, 0); words = list(rng.choice(fam[f], 6)) + list(rng.choice(ham, 2))
    else:
        words = list(rng.choice(ham, 8))
    subj = " ".join(words[:3]); body = " ".join(words) + f" id{i}"
    recv = t.strftime("%a, %d %b %Y %H:%M:%S -0400"); date = "Mon, 01 Jan 1900 00:00:00 +0000" if i % 7 == 0 else recv
    kind = i % 5
    hdr = (f"Received: from mx.example by srv.example with SMTP; {recv}\nReceived: from a by b; Tue, 02 Jan 2001 00:00:00 +0000\n" if i % 23 else "")
    if kind == 0:
        msg = f"{hdr}Date: {date}\nSubject: {subj}\nContent-Type: text/plain; charset=us-ascii\n\n{body}\n"
    elif kind == 1:
        msg = f"{hdr}Date: {date}\nSubject: {subj}\nContent-Type: text/html; charset=us-ascii\n\n<html><style>p{{x}}</style><body><p>{body}</p></body></html>\n"
    elif kind == 2:
        b64 = base64.b64encode(body.encode()).decode()
        msg = (f"{hdr}Date: {date}\nSubject: {subj}\nMIME-Version: 1.0\nContent-Type: multipart/alternative; boundary=XX\n\n--XX\n"
               f"Content-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: base64\n\n{b64}\n--XX--\n")
    elif kind == 3:
        enc = "=?iso-8859-1?q?" + subj.replace(" ", "_") + "_caf=E9?="
        msg = f"{hdr}Date: {date}\nSubject: {enc}\nContent-Type: text/plain; charset=iso-8859-1\nContent-Transfer-Encoding: 8bit\n\n{body} caf\xe9\n"
    else:
        msg = f"{hdr}Date: {date}\nSubject: {subj}\n\n{body}\n"
    with open(f"{root}/data/inmail.{i + 1}", "wb") as f:
        f.write(msg.encode("latin-1"))
    lines.append(f"{'spam' if spam else 'ham'} ../data/inmail.{i + 1}")
open(f"{root}/full/index", "w").write("\n".join(lines) + "\n")
print("synthetic corpus:", root, n)
