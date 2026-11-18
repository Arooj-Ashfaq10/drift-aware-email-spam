# Drift-aware few-shot spam detection on a real time-ordered email stream (TREC 2007)

Base paper: Spam-T5 (Labonne & Moran, arXiv:2304.01238), reproduced with `reproduce_spamt5.py`.
Own work: prompt-example selection (recent vs random), refresh schedules and a label-free content-drift monitor (DTR) on the
TREC 2007 corpus, whose emails are in real delivery order (8 Apr - 6 Jul 2007; timestamps from the top `Received:` header).

| Step | Script | Hardware |
|---|---|---|
| 1 parse corpus, build 12 periods | `prep_trec.py` | CPU |
| 2 DTR refresh schedules (no labels, no LLM) | `monitor_schedule.py` | CPU |
| 3 classical baseline (TF-IDF + LR) with the same label budgets | `run_classical.py` | CPU |
| 4 LLM few-shot experiments (zero-shot, static, recent/random, periodic, DTR, equal-budget control) | `run_email_drift.py` | GPU |
| 5 base-paper reproduction | `reproduce_spamt5.py` | GPU |
| 6 tables, tests, figures | `analyze.py` | CPU |

`tests/` holds a synthetic-corpus smoke test (mock LLM). The TREC corpus is NOT included (its licence forbids republishing);
download it from the original source and cite Cormack (TREC 2007 Spam Track).
