# Drift-Triggered Demonstration Refresh (DTDR) for few-shot LLM spam filtering

Code and result files for the paper *Drift-Triggered Demonstration Refresh for Label-Efficient Few-Shot LLM Spam Filtering*
(Research Methodology, FAST National University of Computer and Emerging Sciences, Islamabad).

A frozen LLM (Qwen2.5-3B-Instruct) classifies a time-ordered email stream with six labeled demonstrations in its prompt.
A label-free character n-gram monitor watches the unlabeled emails of each week and buys six new labeled demonstrations
only when the input distribution moves more than `m` times the Week-0 noise floor. Base paper: Spam-T5
(Labonne & Moran, arXiv:2304.01238), reproduced independently in `reproduce_spamt5.py`.

## Data

* **TREC 2007 Public Spam Corpus** (75,419 emails, 8 Apr - 6 Jul 2007, ordered by the top `Received:` header, 12 weekly
  periods). The corpus is **not included**: its licence forbids redistribution. Obtain it from the TREC Spam Track and run
  `prep_trec.py`.
* **CEAS-08** from the curated email bundle on Zenodo (record 8339691). It is downloaded by the `ceas` stage in `kaggle/make_stage.py` and converted by
  `prep_ceas.py` (ordered by the sender `Date` header, 12 equal-count periods, because it spans only five days).

No email text is stored in this repository; result files contain email ids, labels, scores and predictions only.

## Pipeline

| Step | Script | Hardware | Paper |
|---|---|---|---|
| Parse TREC, build 12 weekly periods | `prep_trec.py` | CPU | Sec. 3, data table |
| Label-free monitor and refresh schedules | `monitor_schedule.py` | CPU | Monitor, m sweep |
| Classical and incremental baselines (TF-IDF + LR, SGD, PA) | `run_classical.py` | CPU | Table `tab:main` |
| LLM policies: zero-shot, static, scheduled, random, DTDR, active, ablations | `run_email_drift.py --suite main` | GPU | Tables `tab:main`, `tab:abl2` |
| Selection rules and partial replacement | `run_email_drift.py --suite selection partial` | GPU | Table `tab:sel` |
| k, prompt wording, natural class ratio, second LLM | `run_email_drift.py --k / --prompt_variant / --eval_mode natural / --model` | GPU | Table `tab:gen` |
| Fine-tuned DistilBERT baseline | `baseline_distilbert.py` | GPU | Ablations text |
| Controlled drift streams and detector comparison (KS, MMD, Page-Hinkley, DDM) | `synthetic_drift.py` | CPU | Table `tab:det`, Fig. 5b |
| Second corpus CEAS-08 | `prep_ceas.py` + the runners above | GPU | Table `tab:ceas` |
| Shorter monitoring windows | `window_size.py` | CPU | Failure analysis |
| Two-stage filter (classifier + LLM) | `two_stage.py` | CPU | Table `tab:twostage` |
| Metrics, two-level bootstrap CIs, paired tests with Holm, McNemar, calibration | `analyze.py`, `analyze_revision.py` | CPU | Tables `tab:ci`, `tab:paired`, `tab:mcnemar`, `tab:calib` |
| LaTeX tables and figure data | `make_latex10.py` | CPU | all tables and figures |
| Spam-T5 reproduction | `reproduce_spamt5.py` (`colab_email.ipynb`) | GPU | Table `tab:t5` |

All GPU runs were executed as Kaggle script kernels on an NVIDIA T4 (FP16, `--no_4bit`); `kaggle/make_stage.py` builds
the kernel for each stage and lists the exact commands and seeds.

## Results

`results/` holds the outputs behind every number in the paper:

| Folder | Content |
|---|---|
| `trec_main_10seeds/` | main LLM suite, seeds 0-9 |
| `trec_classical/` | classical and incremental baselines, seeds 0-9 |
| `trec_analysis/` | CIs, paired tests, McNemar, calibrated operating points, cost frontier, figure data |
| `trec_detectors/` | controlled-stream detector evaluation |
| `trec_selection_partial/`, `trec_variants/`, `trec_phi35/`, `trec_distilbert/` | additional checks (2-3 seeds) |
| `ceas_llm/`, `ceas_classical/`, `ceas_monitor/` | second corpus |
| `window_size/`, `two_stage/` | CPU analyses for monitoring windows and the two-stage filter |
| `spamt5/`, `spamt5_sa/` | Spam-T5 reproduction |
| `summaries/` | short per-policy summaries |

Each run folder has `predictions.csv` (policy, seed, period, id, label, prediction, score) and `run_meta.csv`
(labels used and inspected, refreshes, time, tokens, peak GPU memory).

## Quick start

```bash
pip install -r requirements.txt
python tests/smoke_test.py                       # synthetic corpus + mock LLM, checks the whole pipeline on CPU
python prep_trec.py --root /path/to/trec07p --out trec07p_parsed.csv.gz
python run_email_drift.py --parsed trec07p_parsed.csv.gz --out results/main --no_4bit --suite main --seeds 0 1 2 3 4 5 6 7 8 9
python run_classical.py --parsed trec07p_parsed.csv.gz --out results/classical --seeds 0 1 2 3 4 5 6 7 8 9
python analyze.py --dirs results/main results/classical --out results/analysis
```

Re-analysis without a GPU is possible from the saved files, for example
`python two_stage.py --llm results/trec_main_10seeds/predictions.csv --clf results/trec_classical/predictions.csv --out out/two_stage`.

## Citation of data

G. V. Cormack, "TREC 2007 Spam Track Overview," in *Proc. TREC*, 2007.
