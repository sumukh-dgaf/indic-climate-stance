"""Round 2 schedule: translate, then every training run in priority order, skipping runs that already finished.

Same command on Colab and on a local CUDA machine:
  python -u src/run_all.py                                   # all runs
  python -u src/run_all.py --results-zip colab_results.zip   # ...then pack the results
A run counts as finished when outputs/runs/<model>_<config>.json exists, so a disconnected session resumes where it
stopped. Only the transformer checkpoint the demo needs (models/indicbert_C_full) is kept, to save disk/Drive space.
Self-check: python src/run_all.py --check
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import synthetic_complete  # noqa: E402

RUNS = [("indicbert", "C_full"), ("indicbert", "A_english"), ("indicbert", "B_translated"),
        ("tfidf", "A_english"), ("tfidf", "B_translated"), ("tfidf", "C_full"),
        ("muril", "C_full"), ("muril", "A_english")]
KEEP_MODEL = "indicbert_C_full"
KEEP = {KEEP_MODEL, "indicbert_B_translated"}  # demo model and its fallback (demo.py)
RESULTS = ["outputs/preds/*.csv", "outputs/runs/*.json", "data/translated/*", "data/review/translation_review_*.csv",
           f"models/{KEEP_MODEL}/*"]


def command(model, config):
    if model == "tfidf":
        return [sys.executable, "src/baseline.py", "--config", config]
    return [sys.executable, "src/train_transformer.py", "--model", model, "--config", config]


def drop_checkpoint(model, name):
    # only once every eval set (incl. the synthetic test) is predicted; the demo's checkpoint is always kept
    if model != "tfidf" and name not in KEEP and os.path.exists(f"outputs/preds/{name}_synthetic_test.csv"):
        shutil.rmtree(f"models/{name}", ignore_errors=True)


def backfill(model, config):
    """A run finished before the synthetic data was complete: predict the synthetic test set (and refresh the rest)."""
    name = f"{model}_{config}"
    print(f"== {name}: done earlier; predicting the now-complete synthetic test set", flush=True)
    if model == "tfidf":
        subprocess.run(command(model, config))  # deterministic refit in seconds, rewrites every prediction file
    elif os.path.isdir(f"models/{name}"):
        subprocess.run([sys.executable, "src/predict.py", "--model-dir", f"models/{name}", "--eval-all", name])
    else:
        print(f"!! models/{name} is gone: delete outputs/runs/{name}.json to retrain it")
    drop_checkpoint(model, name)


def pack(path):
    n = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for pattern in RESULTS:
            for f in sorted(glob.glob(pattern)):
                z.write(f, f.replace(os.sep, "/"))
                n += 1
    print(f"wrote {path} ({n} files, {os.path.getsize(path) / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-zip")
    ap.add_argument("--skip-translate", action="store_true")
    a = ap.parse_args()
    if not a.skip_translate:
        r = subprocess.run([sys.executable, "src/translate.py"])
        if r.returncode:
            print("!! translation failed: B/C runs will fail until data/translated/ exists; A runs continue")
    complete = synthetic_complete()
    if not complete:
        print("!! synthetic data not complete (generation unfinished or interrupted): C runs are SKIPPED, A/B checkpoints "
              "are kept so the synthetic test set can be predicted later. Re-run this script once it is complete.")
    for model, config in RUNS:
        name = f"{model}_{config}"
        if os.path.exists(f"outputs/runs/{name}.json"):
            if complete and not os.path.exists(f"outputs/preds/{name}_synthetic_test.csv"):
                backfill(model, config)
            else:
                print(f"== {name}: already done, skipped")
            continue
        if config == "C_full" and not complete:
            print(f"== {name}: waiting for complete synthetic data, skipped")
            continue
        print(f"\n== {name}: starting {time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        r = subprocess.run(command(model, config))
        print(f"== {name}: {'OK' if r.returncode == 0 else 'FAILED (exit %d)' % r.returncode} in {(time.time() - t0) / 60:.1f} min",
              flush=True)
        drop_checkpoint(model, name)
    print("\nrun                       best val F1   quick test macro-F1 (authoritative numbers: src/evaluate.py)")
    for model, config in RUNS:
        p = f"outputs/runs/{model}_{config}.json"
        if os.path.exists(p):
            r = json.load(open(p))
            print(f"{r['name']:25s} {r['best_val_macro_f1']:.3f}        {r['eval_macro_f1_quick']}")
        else:
            print(f"{model}_{config:19s} not run")
    if a.results_zip:
        pack(a.results_zip)


def selfcheck():
    assert RUNS[0] == ("indicbert", "C_full") and len(set(RUNS)) == len(RUNS)  # priority order, no repeats
    assert command("tfidf", "A_english")[1:] == ["src/baseline.py", "--config", "A_english"]
    assert command("muril", "C_full")[-4:] == ["--model", "muril", "--config", "C_full"]
    assert any(KEEP_MODEL in p for p in RESULTS)
    print("run_all.py self-check passed")


if __name__ == "__main__":
    selfcheck() if sys.argv[1:] == ["--check"] else main()
