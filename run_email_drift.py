"""LLM few-shot experiments on the TREC 2007 (or any parsed) time-ordered stream - v5, resumable.

Suites (--suite, several allowed):
  main       zs, static, recent/random every 1 and 3 weeks, DTR(m), equal-budget even schedule, fixed refresh after
             Week 1, random-time trigger, DTR without reference reset, DTR with guards, uncertainty (active) selection
             at the DTR budget                                                     (comments 29, 35, 43, 44, 51)
  selection  random / newest / similarity / diversity / uncertainty selection under static, even and DTR timing (24)
  partial    DTR keeping 2 or 3 old demonstrations instead of replacing all six     (22)
Other knobs: --k (25), --prompt_variant (31), --eval_mode natural (15), --model (36), --seeds 0..9 (44).
Every run records labels used AND labels inspected (26/27), tokens, peak GPU memory and ms/email (54)."""
import argparse, json, os, time
import numpy as np, pandas as pd
from src.common import build_system_prompt
from src.email_data import build_email_stream, stream_summary
from src.llm import CachedClassifier, make_backend
from src.strategies import (select, keep_closest, centroid, window, history, period_stats, dtr_schedule,
                            periodic_schedule, even_schedule, random_schedule)


def run_policy(clf, stream, spec, k, seed, variant, eval_col):
    P = int(stream.period.max()) + 1; how = spec.get("how", "random"); src = spec.get("source", "recent")
    sched = spec["schedule"](seed) if callable(spec["schedule"]) else spec["schedule"]
    scorer = lambda ex: (lambda texts: clf.score(build_system_prompt(ex, variant), texts))
    ex, inspected, used = [], 0, 0
    if spec["name"] != "zs":
        w0 = window(stream, 0)
        ex, ins, u = select(w0, k, seed * 1000, how, ref_centroid=centroid(w0.ex_text), scorer=scorer([]))
        inspected += ins; used += u
    refreshes, rows = 0, []
    for t in range(P):
        refreshed = False
        if spec["name"] != "zs" and t >= 2 and t in sched:
            pool = window(stream, t - 1) if src == "recent" else history(stream, t)
            ref = centroid(window(stream, t - 1).ex_text)
            kept = keep_closest(ex, ref, spec.get("keep", 0))
            ex, ins, u = select(pool, k, seed * 1000 + t, how, keep=kept, ref_centroid=ref, scorer=scorer(ex))
            inspected += ins; used += u; refreshed = True; refreshes += 1
        ev = stream[(stream.period == t) & stream[eval_col]]
        sc = clf.score(build_system_prompt(ex, variant), ev.msg_text.tolist()); pred = (sc > 0).astype(int)
        rows += [dict(policy=spec["name"], seed=seed, period=t, id=int(i), y=int(y), pred=int(p), score=float(s), refreshed=refreshed)
                 for i, y, p, s in zip(ev.id, ev.label, pred, sc)]
    return pd.DataFrame(rows), dict(policy=spec["name"], seed=seed, refreshes=refreshes, labels_used=used, labels_inspected=inspected,
                                    schedule=" ".join(map(str, sorted(sched))))


def build_specs(a, T, cents, noise):
    specs, log = [], {}
    add = lambda **s: specs.append(s)
    if "main" in a.suite:
        add(name="zs", schedule=[]); add(name="static", schedule=[])
        for R in a.intervals:
            add(name=f"recent_R{R}", schedule=periodic_schedule(T, R)); add(name=f"random_R{R}", source="random", schedule=periodic_schedule(T, R))
        add(name="fixed_w1", schedule=[2])
        for m in a.dtr_mult:
            sch, traj = dtr_schedule(cents, noise, m); n = len(sch); log[f"dtr_m{m:g}"] = dict(refresh=sch, trajectory=traj)
            add(name=f"dtr_m{m:g}", schedule=sch)
            add(name=f"recent_even{n}_m{m:g}", schedule=even_schedule(T, n))
            add(name=f"randtrig_m{m:g}", schedule=(lambda s, n=n: random_schedule(T, n, s)))
            add(name=f"active_even{n}_m{m:g}", schedule=even_schedule(T, n), how="uncertainty")
            add(name=f"active_dtr_m{m:g}", schedule=sch, how="uncertainty")
            nr, _ = dtr_schedule(cents, noise, m, reset=False); add(name=f"dtr_noreset_m{m:g}", schedule=nr)
            gd, _ = dtr_schedule(cents, noise, m, cooldown=2, persistence=2); add(name=f"dtr_guard_m{m:g}", schedule=gd)
            log[f"dtr_noreset_m{m:g}"] = nr; log[f"dtr_guard_m{m:g}"] = gd
    if "selection" in a.suite:
        m = a.dtr_mult[0]; sch, _ = dtr_schedule(cents, noise, m); ev = even_schedule(T, len(sch))
        for how in ("random", "newest", "similarity", "diversity", "uncertainty"):
            add(name=f"sel_{how}_static", schedule=[], how=how)
            add(name=f"sel_{how}_even", schedule=ev, how=how)
            add(name=f"sel_{how}_dtr", schedule=sch, how=how)
    if "partial" in a.suite:
        for m in a.dtr_mult:
            sch, _ = dtr_schedule(cents, noise, m)
            for kp in (2, 3):
                add(name=f"dtr_keep{kp}_m{m:g}", schedule=sch, keep=kp)
    return specs, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--periods", type=int, default=12); ap.add_argument("--eval_per_class", type=int, default=50)
    ap.add_argument("--eval_mode", choices=["balanced", "natural"], default="balanced"); ap.add_argument("--natural_per_period", type=int, default=300)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--ex_chars", type=int, default=300); ap.add_argument("--msg_chars", type=int, default=500)
    ap.add_argument("--prompt_variant", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--intervals", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--dtr_mult", type=float, nargs="+", default=[6, 8])
    ap.add_argument("--suite", nargs="+", default=["main"], choices=["main", "selection", "partial"])
    ap.add_argument("--only", nargs="+", default=None, help="run only these policy names")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    stream = build_email_stream(a.parsed, a.periods, a.eval_per_class, ex_chars=a.ex_chars, msg_chars=a.msg_chars,
                                natural_per_period=a.natural_per_period if a.eval_mode == "natural" else 0)
    eval_col = "evalnat" if a.eval_mode == "natural" else "evalset"
    stream[["id", "label", "recv", "period", "split", "evalset", "evalnat"]].to_csv(os.path.join(a.out, "stream_meta.csv"), index=False)
    stream_summary(stream).to_csv(os.path.join(a.out, "stream_summary.csv")); T = a.periods
    cents, noise = period_stats(stream)
    specs, log = build_specs(a, T, cents, noise)
    json.dump(log, open(os.path.join(a.out, "dtr_schedules_used.json"), "w"), indent=1, default=float)
    if a.only:
        specs = [s for s in specs if any(s["name"] == o or (o.endswith("*") and s["name"].startswith(o[:-1])) for o in a.only)]
    pred_path, meta_path = os.path.join(a.out, "predictions.csv"), os.path.join(a.out, "run_meta.csv")
    done = set()
    if os.path.exists(meta_path):
        m0 = pd.read_csv(meta_path); done = set(zip(m0.policy, m0.seed))
    clf = CachedClassifier(make_backend(a.backend, a.model, not a.no_4bit),
                           cache_path=os.path.join(a.out, f"cache_{a.backend}_{a.model.replace('/', '_')}.pkl"), batch_size=a.batch_size)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=2)
    t_all = time.time()
    for spec in specs:
        seeds = [0] if spec["name"] == "zs" else a.seeds
        for s in seeds:
            if (spec["name"], s) in done:
                continue
            t0 = time.time(); p, m = run_policy(clf, stream, spec, a.k, s, a.prompt_variant, eval_col)
            be = clf.backend
            m.update(k=a.k, prompt_variant=a.prompt_variant, eval_mode=a.eval_mode, model=a.model, seconds=round(time.time() - t0, 1),
                     ms_per_email_uncached=round(clf.ms_per_msg, 1), tokens_total=getattr(be, "tokens", 0), peak_gpu_gb=round(getattr(be, "peak_mem_gb", 0.0), 2))
            p.to_csv(pred_path, mode="a", header=not os.path.exists(pred_path), index=False)
            pd.DataFrame([m]).to_csv(meta_path, mode="a", header=not os.path.exists(meta_path), index=False)
            print(f"done {spec['name']} seed={s} refreshes={m['refreshes']} used={m['labels_used']} inspected={m['labels_inspected']} | {m['seconds']}s", flush=True)
    print(f"all done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
