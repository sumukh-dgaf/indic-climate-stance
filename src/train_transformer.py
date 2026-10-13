"""Fine-tune a multilingual encoder for target-aware stance classification (WRITTEN, NOT RUN in Milestone 1).

Input is a PAIR (claim, comment): stance is defined relative to the claim, so the model must see both.
The tokenizer turns the pair into "[CLS] claim [SEP] comment [SEP]" (XLM-R: "<s> claim </s></s> comment </s>").

CPU smoke test :  python src/train_transformer.py --smoke
Colab GPU run  :  pip install transformers torch sentencepiece
                  python src/train_transformer.py --model muril --epochs 4
Other encoders :  --model indic-bert | --model xlm-r
Selection uses macro-F1 on a held-out English split; the Indic pilot is only REPORTED (never used to pick a checkpoint).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLAIM, LABELS, load, load_english, validate  # noqa: E402
from preprocess import clean  # noqa: E402

MODELS = {"muril": "google/muril-base-cased", "indic-bert": "ai4bharat/indic-bert", "xlm-r": "xlm-roberta-base"}
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}


def pair_text(comment):
    """Human-readable view of the target-aware input (the tokenizer builds the real pair)."""
    return f"{CLAIM} [SEP] {comment}"


def class_weights(ids, k=3):
    """'balanced' weights n / (k * n_class): the rare Against class gets the largest loss weight."""
    counts = np.bincount(ids, minlength=k)
    return len(ids) / (k * np.maximum(counts, 1))


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tok, comments, max_len):
    return tok([CLAIM] * len(comments), list(comments), truncation="only_second", max_length=max_len,
               padding="max_length", return_tensors="pt")


def predict_ids(model, enc, device, batch=32):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, enc["input_ids"].shape[0], batch):
            b = {k: v[i:i + batch].to(device) for k, v in enc.items()}
            out.append(model(**b).logits.argmax(-1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="muril")
    ap.add_argument("--train", default="data/english_stance_merged.csv")
    ap.add_argument("--report-on", default="data/pilot_indic_stance.csv", help="reported each epoch, never used for selection")
    ap.add_argument("--out", default="outputs/models/muril")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: 64 train rows, 1 epoch, max_len 48")
    a = ap.parse_args()

    import torch
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_english(a.train) if "merged" in a.train else validate(load(a.train), a.train)
    rep = validate(load(a.report_on, default_source="team-authored pilot (synthetic, dev only)"), a.report_on)
    if a.smoke:
        df = df.sample(64, random_state=a.seed)
        rep = rep.sample(24, random_state=a.seed)
        a.epochs, a.max_len, a.batch = 1, 48, 8
    tr, va = train_test_split(df, test_size=0.1, stratify=df.stance, random_state=a.seed)
    print(f"device={device} model={MODELS[a.model]} train={len(tr)} val={len(va)} report={len(rep)}")
    print("example input:", pair_text(clean(tr.comment.iloc[0]))[:160])

    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    net = AutoModelForSequenceClassification.from_pretrained(MODELS[a.model], num_labels=3).to(device)
    enc = lambda d: encode(tok, d.comment.map(clean), a.max_len)  # noqa: E731
    Xtr, Xva, Xrep = enc(tr), enc(va), enc(rep)
    ytr = torch.tensor(tr.stance.map(LABEL2ID).values)
    yva = va.stance.map(LABEL2ID).values
    yrep = rep.stance.map(LABEL2ID).values

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights(ytr.numpy()), dtype=torch.float, device=device))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best = -1.0
    log = []
    for ep in range(1, a.epochs + 1):
        net.train()
        order = torch.randperm(len(tr))
        for i in range(0, len(tr), a.batch):
            idx = order[i:i + a.batch]
            b = {k: v[idx].to(device) for k, v in Xtr.items()}
            loss = loss_fn(net(**b).logits, ytr[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        f1_val = f1_score(yva, predict_ids(net, Xva, device), average="macro", labels=[0, 1, 2], zero_division=0)
        f1_rep = f1_score(yrep, predict_ids(net, Xrep, device), average="macro", labels=[0, 1, 2], zero_division=0)
        log.append({"epoch": ep, "val_macro_f1_english": float(f1_val), "pilot_macro_f1_dev_only": float(f1_rep)})
        print(f"epoch {ep}: English val macro-F1 {f1_val:.3f} | pilot (dev only, not official) macro-F1 {f1_rep:.3f}")
        if f1_val > best:  # selection by macro-F1 on English validation only
            best = f1_val
            os.makedirs(a.out, exist_ok=True)
            net.save_pretrained(a.out)
            tok.save_pretrained(a.out)
            json.dump({"labels": LABELS, "max_len": a.max_len, "base_model": MODELS[a.model], "seed": a.seed, "log": log},
                      open(os.path.join(a.out, "run_info.json"), "w"), indent=1)
    print(f"best English validation macro-F1 {best:.3f}; checkpoint in {a.out}")


def selfcheck():
    assert pair_text("x") == CLAIM + " [SEP] x"
    w = class_weights(np.array([0, 0, 0, 1, 2, 2]))
    assert np.allclose(w, [6 / 9, 6 / 3, 6 / 6]) and w.argmax() == 1  # singleton class gets the biggest weight
    assert LABEL2ID == {"Favour": 0, "Against": 1, "None": 2}
    assert MODELS["muril"] == "google/muril-base-cased"
    print("train_transformer.py self-check passed (torch not required for this check)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selfcheck()
        print("usage: python src/train_transformer.py --smoke | --model {muril,indic-bert,xlm-r}")
    else:
        main()
