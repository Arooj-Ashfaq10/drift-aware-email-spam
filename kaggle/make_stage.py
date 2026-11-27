"""Build a Kaggle script-kernel folder for one stage and print the push command.

usage: python make_stage.py <stage> [--gpu/--cpu]
Each stage runs the v5 code from the private dataset aroojashfaq/trec07p-drift and writes everything to /kaggle/working/out,
which is downloaded afterwards with `kaggle kernels output`."""
import json, os, sys

STAGES = {
    # name: (gpu, list of shell commands run inside the code folder; $P = parsed csv, $O = output root)
    "pilot": (True, [
        "nvidia-smi --query-gpu=name,memory.total --format=csv",
        "python run_email_drift.py --parsed $P --out $O/pilot --no_4bit --seeds 0 --only zs static dtr_m8 active_even1_m8",
    ]),
    "main_a": (True, ["python run_email_drift.py --parsed $P --out $O/main --no_4bit --suite main --seeds 0 1 2 3 4"]),
    "main_b": (True, ["python run_email_drift.py --parsed $P --out $O/main --no_4bit --suite main --seeds 5 6 7 8 9"]),
    "select": (True, ["python run_email_drift.py --parsed $P --out $O/selection --no_4bit --suite selection partial --dtr_mult 8 6 --seeds 0 1 2"]),
    "variants": (True, [
        "python run_email_drift.py --parsed $P --out $O/natural --no_4bit --eval_mode natural --natural_per_period 300 --seeds 0 1 --only zs static recent_R1 dtr_m8 recent_even*",
        *[f"python run_email_drift.py --parsed $P --out $O/k{k} --no_4bit --k {k} --seeds 0 1 --only static dtr_m8 recent_even*" for k in (2, 4, 8, 12)],
        *[f"python run_email_drift.py --parsed $P --out $O/prompt{v} --no_4bit --prompt_variant {v} --seeds 0 1 2 --only zs static dtr_m8" for v in (1, 2)],
    ]),
    "llm2": (True, ["python run_email_drift.py --parsed $P --out $O/phi35 --no_4bit --model microsoft/Phi-3.5-mini-instruct --seeds 0 1 2 --only zs static recent_R1 dtr_m8 recent_even* fixed_w1"]),
    "cpu": (False, [
        "python run_classical.py --parsed $P --out $O/classical --seeds 0 1 2 3 4 5 6 7 8 9",
        "pip -q install sentence-transformers || true",
        "python synthetic_drift.py --parsed $P --out $O/synthetic --seeds 0 1 2 3 4",
    ]),
    "ceas": (True, [
        "python -c \"import requests,subprocess; r=requests.get('https://zenodo.org/api/records/8339691',timeout=60).json(); "
        "u=[f['links']['self'] for f in r['files'] if f['key']=='CEAS_08.csv'][0]; subprocess.run(['wget','-q','-O','/kaggle/working/CEAS_08.csv',u],check=True)\"",
        "python prep_ceas.py --csv /kaggle/working/CEAS_08.csv --out /kaggle/working/ceas08_parsed.csv.gz",
        "python monitor_schedule.py --parsed /kaggle/working/ceas08_parsed.csv.gz --out $O/ceas_monitor",
        "python run_classical.py --parsed /kaggle/working/ceas08_parsed.csv.gz --out $O/ceas_classical --budgets 6 50 500 --seeds 0 1 2",
        "python run_email_drift.py --parsed /kaggle/working/ceas08_parsed.csv.gz --out $O/ceas_llm --no_4bit --seeds 0 1 2 --dtr_mult 6 8 "
        "--only zs static recent_R1 random_R1 fixed_w1 dtr_m6 dtr_m8 recent_even*",
    ]),
    "bert": (True, ["python baseline_distilbert.py --parsed $P --out $O/distilbert --budgets 50 500 --seeds 0 1 2"]),
}

RUNNER = r'''import glob, os, shutil, subprocess, sys, time
src = sorted(glob.glob("/kaggle/input/**/run_email_drift.py", recursive=True), key=lambda x: "dtdr-code" not in x)
assert src, os.listdir("/kaggle/input")
CODE = "/kaggle/working/code"; shutil.copytree(os.path.dirname(src[0]), CODE, dirs_exist_ok=True)
P = [p for p in glob.glob("/kaggle/input/**/trec07p_parsed*", recursive=True) if os.path.isfile(p)][0]
O = "/kaggle/working/out"; os.makedirs(O, exist_ok=True)
print("code:", src[0], "| parsed:", P, flush=True)
os.chdir(CODE)
for cmd in CMDS:
    cmd = cmd.replace("$P", P).replace("$O", O); t0 = time.time(); print("\n>>>", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    print(f"<<< exit {r.returncode} after {(time.time() - t0) / 60:.1f} min", flush=True)
for f in glob.glob(O + "/**/*.pkl", recursive=True):
    os.remove(f)                                  # caches are large and only needed inside one run
print("STAGE DONE", flush=True)
'''


def main():
    stage = sys.argv[1]; gpu, cmds = STAGES[stage]
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), stage); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "run.py"), "w").write(f"CMDS = {json.dumps(cmds, indent=1)}\n" + RUNNER)
    meta = dict(id=f"aroojashfaq/dtdr-{stage.replace('_', '-')}", title=f"dtdr-{stage.replace('_', '-')}", code_file="run.py",
                language="python", kernel_type="script", is_private=True, enable_gpu=gpu, enable_internet=True,
                dataset_sources=["aroojashfaq/trec07p-drift", "aroojashfaq/dtdr-code"], competition_sources=[], kernel_sources=[])
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    print(d)


if __name__ == "__main__":
    main()
