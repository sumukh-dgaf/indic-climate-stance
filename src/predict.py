"""Inference shared by training, re-prediction and the demo.

Submission file   : python src/predict.py --model-dir models/indicbert_C_full --input test.csv --output outputs/submission.csv
Re-predict all    : python src/predict.py --model-dir models/indicbert_C_full --eval-all indicbert_C_full
                    (writes outputs/preds/indicbert_C_full_<evalset>.csv, e.g. after a session died before predicting)
The input needs a text column and (ideally) an id column; names are sniffed by data.load.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLAIM, LABELS, eval_sets, load, write_preds  # noqa: E402
from preprocess import clean  # noqa: E402


def predict_probs(net, tok, comments, max_len, device, batch=64):
    """(n, 3) softmax probabilities in LABELS order for (claim, comment) pairs; the claim stays English."""
    import torch
    net.eval()
    texts = [clean(c) for c in comments]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok([CLAIM] * len(chunk), chunk, truncation="only_second", max_length=max_len, padding=True,
                      return_tensors="pt").to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = net(**enc).logits
            out.append(torch.softmax(logits.float(), -1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 3))


def load_model(model_dir, device="cpu"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    info = json.load(open(os.path.join(model_dir, "run_info.json")))
    assert info["labels"] == LABELS, "label order of the checkpoint differs from data.LABELS"
    tok = AutoTokenizer.from_pretrained(model_dir)
    net = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    return tok, net, info


def write_submission(ids, labels, path):
    assert len(ids) == len(labels) and set(labels) <= set(LABELS)
    pd.DataFrame({"ID": list(ids), "Predicted Label": list(labels)}).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input")
    ap.add_argument("--output", default="outputs/submission.csv")
    ap.add_argument("--eval-all", metavar="NAME", help="predict every evaluation set into outputs/preds/NAME_<set>.csv")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, net, info = load_model(a.model_dir, device)
    if a.eval_all:
        for name, df in eval_sets().items():
            write_preds(df, predict_probs(net, tok, df.comment, info["max_len"], device), f"outputs/preds/{a.eval_all}_{name}.csv")
            print(f"wrote outputs/preds/{a.eval_all}_{name}.csv ({len(df)} rows)")
        return
    df = load(a.input, need_label=False)
    probs = predict_probs(net, tok, df.comment, info["max_len"], device)
    write_submission(df["id"], [LABELS[i] for i in probs.argmax(1)], a.output)
    print(f"wrote {len(df)} predictions to {a.output}")


def selfcheck():
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "s.csv")
    write_submission(["a1", "a2", "a3"], ["Favour", "Against", "None"], p)
    assert open(p, encoding="utf-8").readline().strip() == "ID,Predicted Label"
    back = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert back["Predicted Label"].tolist() == ["Favour", "Against", "None"]  # "None" survives the round trip
    print("predict.py self-check passed (torch not required for this check)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selfcheck()
    else:
        main()
