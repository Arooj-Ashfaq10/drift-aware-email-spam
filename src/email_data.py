"""TREC 2007 public corpus (trec07p): parsing and construction of a time-ordered email stream.
The corpus is ordered by delivery time (index order); real timestamps come from the top Received header
(added by the receiving server), NOT from the sender-controlled Date header (often forged)."""
import email, email.header, email.utils, html, os, re
import numpy as np
import pandas as pd

_STYLE = re.compile(r"(?is)<(script|style).*?</\1>")
_TAG = re.compile(r"<[^>]+>")


def _payload(part):
    raw = part.get_payload(decode=True)
    if raw is None:
        return ""
    cs = part.get_content_charset() or "latin-1"
    try:
        return raw.decode(cs, errors="replace")
    except LookupError:
        return raw.decode("latin-1", errors="replace")


def extract_text(msg):
    plain, htmls = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        c = part.get_content_type()
        if c == "text/plain":
            plain.append(_payload(part))
        elif c == "text/html":
            htmls.append(_payload(part))
    if plain:
        txt = "\n".join(plain)
    elif htmls:
        txt = html.unescape(_TAG.sub(" ", _STYLE.sub(" ", "\n".join(htmls))))
    else:
        txt = ""
    return " ".join(txt.split())


def decode_subject(msg):
    s = msg.get("Subject", "") or ""
    try:
        return " ".join(str(email.header.make_header(email.header.decode_header(s))).split())
    except Exception:
        return " ".join(str(s).split())


def top_received_time(msg):
    rec = msg.get_all("Received") or []
    if not rec:
        return pd.NaT
    try:
        ts = pd.Timestamp(email.utils.parsedate_to_datetime(rec[0].rsplit(";", 1)[-1].strip()))
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        return pd.NaT


def parse_corpus(root, out_csv, max_chars=1500, limit=None, log_every=10000):
    """root contains data/ and full/index. Writes CSV: idx,label(1=spam),recv(UTC),subject,body."""
    rows = []
    with open(os.path.join(root, "full", "index")) as f:
        for i, ln in enumerate(f):
            if limit and i >= limit:
                break
            lab, rel = ln.split()
            path = os.path.normpath(os.path.join(root, "full", rel))
            try:
                with open(path, "rb") as fh:
                    msg = email.message_from_binary_file(fh)
                rows.append((i, 1 if lab == "spam" else 0, top_received_time(msg), decode_subject(msg)[:300],
                             extract_text(msg)[:max_chars]))
            except Exception:
                rows.append((i, 1 if lab == "spam" else 0, pd.NaT, "", ""))
            if log_every and (i + 1) % log_every == 0:
                print(f"parsed {i + 1}", flush=True)
    df = pd.DataFrame(rows, columns=["idx", "label", "recv", "subject", "body"])
    df["recv"] = pd.to_datetime(df.recv, utc=True)
    sec = (df.recv - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    n_missing = int(sec.isna().sum())
    sec = sec.interpolate(limit_direction="both")           # corpus is in delivery order -> interpolate gaps by position
    df["recv"] = pd.to_datetime(sec.round(), unit="s", utc=True)          # whole seconds -> one consistent text format
    df.to_csv(out_csv, index=False)
    print(f"saved {len(df)} emails -> {out_csv} | missing Received times filled by interpolation: {n_missing}")
    return df


def build_email_stream(csv_path, n_periods=12, eval_per_class=50, pool_frac0=0.6, seed=0,
                       ex_chars=300, msg_chars=500, natural_per_period=0):
    """Equal-duration periods over the real delivery time span. Period 0: 60% labelled pool / 40% stable eval set.
    evalset = fixed, class-balanced random sample (eval_per_class spam + eval_per_class ham) per period.
    evalnat = fixed uniform random sample of natural_per_period emails per period, keeping the true class ratio (comment 15)."""
    from .common import format_email
    df = pd.read_csv(csv_path)
    df["recv"] = pd.to_datetime(df.recv, utc=True, format="ISO8601")
    df[["subject", "body"]] = df[["subject", "body"]].fillna("").astype(str)
    t0, t1 = df.recv.min(), df.recv.max()
    frac = (df.recv - t0) / (t1 - t0 + pd.Timedelta(seconds=1))
    df["period"] = np.minimum((frac * n_periods).astype(int), n_periods - 1)
    df["id"] = df.idx
    rng = np.random.default_rng(seed)
    df["split"] = "eval"
    for y in (0, 1):
        ix = df.index[(df.period == 0) & (df.label == y)].to_numpy(); ix = rng.permutation(ix)
        df.loc[ix[: int(pool_frac0 * len(ix))], "split"] = "pool"
    df["evalset"] = False
    for p in range(n_periods):
        for y in (0, 1):
            ix = df.index[(df.period == p) & (df.label == y) & ((df.period > 0) | (df.split == "eval"))].to_numpy()
            if len(ix) < eval_per_class:
                raise ValueError(f"period {p}, label {y}: only {len(ix)} eligible emails (< {eval_per_class})")
            df.loc[rng.choice(ix, eval_per_class, replace=False), "evalset"] = True
    df["evalnat"] = False
    if natural_per_period:
        rng2 = np.random.default_rng(seed + 1)
        for p in range(n_periods):
            ix = df.index[(df.period == p) & ((df.period > 0) | (df.split == "eval"))].to_numpy()
            df.loc[rng2.choice(ix, min(natural_per_period, len(ix)), replace=False), "evalnat"] = True
    df["ex_text"] = [format_email(s, b, ex_chars) for s, b in zip(df.subject, df.body)]
    df["msg_text"] = [format_email(s, b, msg_chars) for s, b in zip(df.subject, df.body)]
    return df.reset_index(drop=True)


def stream_summary(df):
    g = df.groupby("period").agg(start=("recv", "min"), end=("recv", "max"), emails=("id", "size"), spam=("label", "sum"))
    g["ham"] = g.emails - g.spam; g["evaluated"] = df[df.evalset].groupby("period").size()
    return g
