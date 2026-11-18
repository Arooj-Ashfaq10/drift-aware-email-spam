"""Step 4/5: LLM few-shot experiments on the TREC 2007 time-ordered stream (resumable)."""
import argparse, json, os, time
import numpy as np, pandas as pd
from src.common import build_system_prompt
from src.email_data import build_email_stream, stream_summary
from src.llm import CachedClassifier, make_backend
from src.strategies import pick_examples, window, history, period_stats, dtr_schedule, periodic_schedule, even_schedule


def run_policy(clf, stream, name, source, refresh, k, seed):
    P = int(stream.period.max()) + 1
    ex = [] if name == "zs" else pick_examples(window(stream, 0), k, seed)
    refreshes, rows = 0, []
    for t in range(P):
        refreshed = False
        if name != "zs" and t >= 2 and t in refresh:
            pool = window(stream, t - 1) if source == "recent" else history(stream, t)
            ex = pick_examples(pool, k, seed * 1000 + t); refreshed = True; refreshes += 1
        ev = stream[(stream.period == t) & stream.evalset]
        sc = clf.score(build_system_prompt(ex), ev.msg_text.tolist())
        pred = (sc > 0).astype(int)
        for mid, y, p_, s_ in zip(ev.id, ev.label, pred, sc):
            rows.append(dict(policy=name, seed=seed, period=t, id=int(mid), y=int(y), pred=int(p_), score=float(s_), refreshed=refreshed))
    labels = 0 if name == "zs" else k * (1 + refreshes)
    return pd.DataFrame(rows), dict(policy=name, seed=seed, refreshes=refreshes, labels_used=labels, schedule=" ".join(map(str, sorted(refresh))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--periods", type=int, default=12); ap.add_argument("--eval_per_class", type=int, default=50)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--ex_chars", type=int, default=300); ap.add_argument("--msg_chars", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--intervals", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--dtr_mult", type=float, nargs="+", default=[2, 4, 8])
    ap.add_argument("--only", nargs="+", default=None, help="run only these policy names (pilot)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    stream = build_email_stream(a.parsed, a.periods, a.eval_per_class, ex_chars=a.ex_chars, msg_chars=a.msg_chars)
    stream[["id", "label", "recv", "period", "split", "evalset"]].to_csv(os.path.join(a.out, "stream_meta.csv"), index=False)
    stream_summary(stream).to_csv(os.path.join(a.out, "stream_summary.csv")); T = a.periods
    cents, noise = period_stats(stream)
    specs = [("zs", "recent", [], [0]), ("static", "recent", [], a.seeds)]
    for R in a.intervals:
        specs += [(f"recent_R{R}", "recent", periodic_schedule(T, R), a.seeds), (f"random_R{R}", "random", periodic_schedule(T, R), a.seeds)]
    seen = set(); sched_log = {}
    for m in a.dtr_mult:
        sch, traj = dtr_schedule(cents, noise, m); sched_log[f"dtr_m{m:g}"] = dict(refresh=sch, trajectory=[(p, round(d, 4), round(t, 4)) for p, d, t in traj])
        specs.append((f"dtr_m{m:g}", "recent", sch, a.seeds))
        if sch and len(sch) not in seen:
            seen.add(len(sch)); specs.append((f"recent_even{len(sch)}", "recent", even_schedule(T, len(sch)), a.seeds))
    json.dump(sched_log, open(os.path.join(a.out, "dtr_schedules_used.json"), "w"), indent=2)
    if a.only:
        specs = [s for s in specs if s[0] in a.only]
    pred_path, meta_path = os.path.join(a.out, "predictions.csv"), os.path.join(a.out, "run_meta.csv")
    done = set()
    if os.path.exists(meta_path):
        m0 = pd.read_csv(meta_path); done = set(zip(m0.policy, m0.seed))
    clf = CachedClassifier(make_backend(a.backend, a.model, not a.no_4bit),
                           cache_path=os.path.join(a.out, f"cache_{a.backend}_{a.model.replace('/', '_')}.pkl"), batch_size=a.batch_size)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=2)
    t_all = time.time()
    for name, source, refresh, seeds in specs:
        for s in seeds:
            if (name, s) in done:
                print(f"skip (done) {name} seed={s}"); continue
            t0 = time.time(); p, m = run_policy(clf, stream, name, source, refresh, a.k, s)
            p.to_csv(pred_path, mode="a", header=not os.path.exists(pred_path), index=False)
            pd.DataFrame([m]).to_csv(meta_path, mode="a", header=not os.path.exists(meta_path), index=False)
            print(f"done {name} seed={s} refreshes={m['refreshes']} | {time.time() - t0:.0f}s | uncached ms/msg={clf.ms_per_msg:.0f} | cache hits={clf.n_hits}", flush=True)
    print(f"all done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
