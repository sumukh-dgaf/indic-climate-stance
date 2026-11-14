"""Fine-tune a multilingual encoder for target-aware stance on a named data configuration, then predict every eval set.

Input is a PAIR (claim, comment): stance is defined relative to the claim, so the model must see both. The claim
stays in English for every language. The tokenizer builds "[CLS] claim [SEP] comment [SEP]".

  python src/train_transformer.py --model indicbert --config C_full     # GPU (Colab T4 or a local CUDA card)
  python src/train_transformer.py --smoke                               # whole path on CPU with a few rows
Models : indicbert = ai4bharat/IndicBERTv2-MLM-only (main), muril = google/muril-base-cased (comparison)
Configs: A_english (zero-shot transfer) | B_translated (+ machine-translated hi/bn) | C_full (+ synthetic hi/bn)
Selection: best macro-F1 on the validation data of the SAME configuration. Test sets are only predicted, never used to select.
Outputs: models/<model>_<config>/ (best checkpoint only), outputs/runs/<model>_<config>.json,
         outputs/preds/<model>_<config>_<evalset>.csv   (smoke runs write under outputs/smoke/ instead)
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLAIM, CONFIGS, LABELS, SYNTHETIC, config_data, eval_sets, write_preds  # noqa: E402
from predict import predict_probs  # noqa: E402
from preprocess import clean  # noqa: E402

MODELS = {"indicbert": "ai4bharat/IndicBERTv2-MLM-only", "muril": "google/muril-base-cased", "xlm-r": "xlm-roberta-base"}
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
    return tok([CLAIM] * len(comments), [clean(c) for c in comments], truncation="only_second", max_length=max_len,
               padding="max_length", return_tensors="pt")


def macro_f1(gold, probs):
    from sklearn.metrics import f1_score
    return float(f1_score(gold, [LABELS[i] for i in probs.argmax(1)], average="macro", labels=LABELS, zero_division=0))


def synthetic_verified():
    import pandas as pd
    if not os.path.exists(SYNTHETIC):
        return "no synthetic data"
    s = pd.read_csv(SYNTHETIC, dtype=str, keep_default_na=False)
    return "yes" if (s[s.split == "test"].human_verified == "yes").all() else "labels not yet human-verified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="indicbert")
    ap.add_argument("--config", choices=list(CONFIGS), default="C_full")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="CPU, ~48 train rows, 1 epoch, max_len 48, outputs/smoke/")
    a = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(a.seed)
    device = "cpu" if a.smoke or not torch.cuda.is_available() else "cuda"
    name = f"{a.model}_{a.config}"
    root = "outputs/smoke/" if a.smoke else ""
    model_dir, run_path = f"{root}models/{name}", f"{root or 'outputs/'}runs/{name}.json"
    pred_path = lambda ev: f"{root or 'outputs/'}preds/{name}_{ev}.csv"  # noqa: E731

    tr = config_data(a.config, "train", strict=not a.smoke)
    va = config_data(a.config, "val", strict=not a.smoke)
    evs = eval_sets()
    if a.smoke:
        tr = tr.groupby("stance", group_keys=False).apply(lambda g: g.sample(min(16, len(g)), random_state=a.seed))
        va = va.sample(min(24, len(va)), random_state=a.seed)
        evs = {k: v.sample(min(8, len(v)), random_state=a.seed) for k, v in evs.items()}
        a.epochs, a.max_len, a.batch = 1, 48, 8
    print(f"device={device} model={MODELS[a.model]} config={a.config} train={len(tr)} "
          f"{tr.language.value_counts().to_dict()} val={len(va)} eval={ {k: len(v) for k, v in evs.items()} }")
    print("example input:", pair_text(clean(tr.comment.iloc[0]))[:160])

    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    net = AutoModelForSequenceClassification.from_pretrained(MODELS[a.model], num_labels=3).to(device)
    Xtr = encode(tok, tr.comment.tolist(), a.max_len)
    ytr = torch.tensor(tr.stance.map(LABEL2ID).values)
    weights = class_weights(ytr.numpy())
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    amp = device == "cuda"  # mixed precision on GPU only
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, best_state, log, t0 = -1.0, None, [], time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        order = torch.randperm(len(tr))
        for i in range(0, len(tr), a.batch):
            idx = order[i:i + a.batch]
            b = {k: v[idx].to(device) for k, v in Xtr.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                logits = net(**b).logits
            loss = loss_fn(logits.float(), ytr[idx].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
        pv = predict_probs(net, tok, va.comment, a.max_len, device)
        f1 = macro_f1(va.stance, pv)
        by_lang = {l: macro_f1(va.stance[m], pv[m.values]) for l in sorted(va.language.unique()) for m in [va.language == l]}
        log.append({"epoch": ep, "val_macro_f1": round(f1, 4), "val_macro_f1_by_language": {k: round(v, 4) for k, v in by_lang.items()},
                    "last_batch_loss": round(loss.item(), 4), "minutes": round((time.time() - t0) / 60, 2)})
        print(f"epoch {ep}: val macro-F1 {f1:.3f} {by_lang}  ({(time.time() - t0) / 60:.1f} min)")
        if f1 > best:  # selection on this configuration's validation data only
            best = f1
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
    train_min = (time.time() - t0) / 60

    net.load_state_dict(best_state)
    os.makedirs(model_dir, exist_ok=True)
    net.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    json.dump({"labels": LABELS, "max_len": a.max_len, "base_model": MODELS[a.model], "config": a.config, "claim": CLAIM},
              open(os.path.join(model_dir, "run_info.json"), "w"), indent=1)

    quick = {}
    for ev, df in evs.items():
        p = predict_probs(net, tok, df.comment, a.max_len, device)
        write_preds(df, p, pred_path(ev))
        quick[ev] = round(macro_f1(df.stance, p), 4)
    run = {"name": name, "model": a.model, "base_model": MODELS[a.model], "config": a.config, "smoke": a.smoke,
           "train_families": CONFIGS[a.config], "n_train": len(tr), "n_train_by_language": tr.language.value_counts().to_dict(),
           "n_val": len(va), "train_class_counts": tr.stance.value_counts().to_dict(),
           "class_weights": dict(zip(LABELS, np.round(weights, 4).tolist())),
           "settings": {"lr": a.lr, "batch": a.batch, "epochs": a.epochs, "warmup": 0.1, "weight_decay": 0.01,
                        "max_len": a.max_len, "seed": a.seed, "mixed_precision": amp, "loss": "class-weighted cross-entropy"},
           "best_epoch": max(log, key=lambda r: r["val_macro_f1"])["epoch"], "best_val_macro_f1": round(best, 4),
           "epochs_log": log, "training_minutes": round(train_min, 2), "device": device,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "eval_sets": {k: len(v) for k, v in evs.items()}, "eval_macro_f1_quick": quick,
           "synthetic_test_labels": synthetic_verified(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(run_path), exist_ok=True)
    json.dump(run, open(run_path, "w"), indent=1)
    print(f"best val macro-F1 {best:.3f} (epoch {run['best_epoch']}); test-set macro-F1 (quick view, see evaluate.py): {quick}")
    print(f"wrote {model_dir}/, {run_path}, {len(evs)} prediction files")


def selfcheck():
    assert pair_text("x") == CLAIM + " [SEP] x"
    w = class_weights(np.array([0, 0, 0, 1, 2, 2]))
    assert np.allclose(w, [6 / 9, 6 / 3, 6 / 6]) and w.argmax() == 1  # singleton class gets the biggest weight
    assert LABEL2ID == {"Favour": 0, "Against": 1, "None": 2}
    assert MODELS["indicbert"] == "ai4bharat/IndicBERTv2-MLM-only" and "indic-bert" not in MODELS  # not the old v1 model
    assert abs(macro_f1(["Favour", "None"], np.array([[0.9, 0, 0.1], [0.2, 0.1, 0.7]])) - 2 / 3) < 1e-9  # Against absent: F1 0
    print("train_transformer.py self-check passed (torch not required for this check)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selfcheck()
        print("usage: python src/train_transformer.py --smoke | --model {indicbert,muril} --config {A_english,B_translated,C_full}")
    else:
        main()
