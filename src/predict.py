"""Predict stance for an unlabelled CSV and write the submission file "ID,Predicted Label".

python src/predict.py --model-dir outputs/models/muril --input test.csv --output outputs/submission.csv
The input needs a text column and (ideally) an id column; names are sniffed by data.load.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLAIM, LABELS, load  # noqa: E402
from preprocess import clean  # noqa: E402


def write_submission(ids, labels, path):
    assert len(ids) == len(labels) and set(labels) <= set(LABELS)
    pd.DataFrame({"ID": list(ids), "Predicted Label": list(labels)}).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="outputs/submission.csv")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    info = json.load(open(os.path.join(a.model_dir, "run_info.json")))
    df = load(a.input, need_label=False)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    net = AutoModelForSequenceClassification.from_pretrained(a.model_dir).eval()
    texts = df.comment.map(clean).tolist()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok([CLAIM] * len(texts[i:i + 32]), texts[i:i + 32], truncation="only_second",
                      max_length=info["max_len"], padding=True, return_tensors="pt")
            out += net(**enc).logits.argmax(-1).tolist()
    write_submission(df["id"], [info["labels"][i] for i in out], a.output)
    print(f"wrote {len(out)} predictions to {a.output}")


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
