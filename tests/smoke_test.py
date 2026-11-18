"""End-to-end smoke test: synthetic TREC-like corpus + MOCK LLM (no scientific results)."""
import os, subprocess, sys, tempfile
d = tempfile.mkdtemp(); py = sys.executable
steps = [
    [py, "tests/make_synthetic_trec.py", f"{d}/trec07p", "6000"],
    [py, "prep_trec.py", "--root", f"{d}/trec07p", "--out", f"{d}/parsed.csv.gz", "--eval_per_class", "20"],
    [py, "monitor_schedule.py", "--parsed", f"{d}/parsed.csv.gz", "--out", f"{d}/mon", "--eval_per_class", "20"],
    [py, "run_classical.py", "--parsed", f"{d}/parsed.csv.gz", "--out", f"{d}/cls", "--eval_per_class", "20", "--budgets", "6", "--seeds", "0"],
    [py, "run_email_drift.py", "--backend", "mock", "--model", "mock", "--parsed", f"{d}/parsed.csv.gz", "--out", f"{d}/llm", "--eval_per_class", "20", "--seeds", "0"],
    [py, "analyze.py", "--dirs", f"{d}/llm", f"{d}/cls", "--out", f"{d}/ana"],
]
for s in steps:
    r = subprocess.run(s, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-800:], r.stderr[-1500:]); sys.exit(1)
print("SMOKE TEST PASSED")
