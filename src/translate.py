"""Translate the English split files into Hindi and Bengali (runs on Colab GPU; works on any CUDA machine).

Translation happens AFTER the split, so a test sentence and its translation never sit on different sides of it.
Output data/translated/{hi,bn}_{train,val,test}.csv (git-ignored: GWStance has no licence) with the original id kept as
source_id, plus data/review/translation_review_{hi,bn}.csv (100 test rows per language) for the team to check that the
stance label survives translation.

Model: IndicTrans2 distilled (ai4bharat/indictrans2-en-indic-dist-200M) is GATED on Hugging Face (accept its terms and
set HF_TOKEN) and needs `pip install IndicTransToolkit`. Without both, or if it fails, NLLB-200 distilled 600M is used.
The model used is written into every row and into data/translated/translation_info.json.
Run: python src/translate.py [--model auto|indictrans2|nllb]      Self-check only: python src/translate.py --check
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ENG, LABELS, TRANSLATED  # noqa: E402
from preprocess import clean, script_id  # noqa: E402

IT2 = "ai4bharat/indictrans2-en-indic-dist-200M"
NLLB = "facebook/nllb-200-distilled-600M"
TGT = {"hi": "hin_Deva", "bn": "ben_Beng"}  # same codes for both models
REVIEW = "data/review/translation_review_{}.csv"
INFO = "data/translated/translation_info.json"


def load_nllb(device):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLLB, src_lang="eng_Latn")
    net = AutoModelForSeq2SeqLM.from_pretrained(NLLB, dtype=torch.float16 if device == "cuda" else torch.float32)
    net = net.to(device).eval()

    def tr(texts, tgt):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = net.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), num_beams=4, max_length=256)
        return tok.batch_decode(out, skip_special_tokens=True)
    return tr


def load_indictrans2(device):
    import torch
    from IndicTransToolkit.processor import IndicProcessor  # pre/post-processing incl. Bengali script restoration
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    ip = IndicProcessor(inference=True)
    tok = AutoTokenizer.from_pretrained(IT2, trust_remote_code=True)  # official ai4bharat repo ships its own tokenizer code
    net = AutoModelForSeq2SeqLM.from_pretrained(IT2, trust_remote_code=True,
                                                dtype=torch.float16 if device == "cuda" else torch.float32)
    net = net.to(device).eval()

    def tr(texts, tgt):
        batch = ip.preprocess_batch(texts, src_lang="eng_Latn", tgt_lang=tgt)
        enc = tok(batch, truncation=True, padding="longest", max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            out = net.generate(**enc, num_beams=4, max_length=256)
        dec = tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return ip.postprocess_batch(dec, lang=tgt)
    return tr


def pick_translator(choice, device):
    if choice == "indictrans2" or (choice == "auto" and os.environ.get("HF_TOKEN")):
        try:
            tr = load_indictrans2(device)
            probe = tr(["Global warming is a serious problem."], "ben_Beng")
            assert script_id(probe[0]) == "bn", f"unexpected output {probe}"
            return IT2, tr
        except Exception as e:  # gated without access, toolkit missing, remote code changed...
            print(f"IndicTrans2 unavailable ({type(e).__name__}: {str(e)[:200]}) -> falling back to NLLB")
            if choice == "indictrans2":
                raise
    elif choice == "auto":
        print("HF_TOKEN not set: IndicTrans2 is gated -> using NLLB")
    return NLLB, load_nllb(device)


def translate_all(texts, tgt, tr, batch=32):
    """Length-sorted batches (less padding), results returned in the original order."""
    order = np.argsort([len(t) for t in texts])
    out = [None] * len(texts)
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        for j, r in zip(idx, tr([texts[j] for j in idx], tgt)):
            out[j] = r.strip()
    return out


def translate_df(df, lang, tr, model_name):
    hyp = translate_all(df.comment.map(clean).tolist(), TGT[lang], tr)
    return pd.DataFrame({"id": [f"{lang}_{i}" for i in df.id], "source_id": df.id.values, "comment": hyp,
                         "stance": df.stance.values, "language": lang, "split": df.split.values,
                         "source": [f"machine-translated from {s}" for s in df.source],
                         "translation_model": model_name})


def review_sample(en_test, tr_test, n=100, seed=42):
    from sklearn.model_selection import train_test_split
    pick, _ = train_test_split(tr_test, train_size=n, stratify=tr_test.stance, random_state=seed)
    english = dict(zip(en_test.id, en_test.comment))
    return pd.DataFrame({"source_id": pick.source_id.values, "stance": pick.stance.values,
                         "english": pick.source_id.map(english).values, "translation": pick.comment.values,
                         "reviewer": "", "label_survives": "", "notes": ""})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["auto", "indictrans2", "nllb"], default="auto")
    a = ap.parse_args()
    todo = [(lang, s) for lang in TGT for s in ("train", "val", "test") if not os.path.exists(TRANSLATED.format(lang, s))]
    if not todo:
        print("all translated files exist; delete them to re-translate")
        return
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name, tr = pick_translator(a.model, device)
    print(f"translating {todo} with {model_name} on {device}")
    os.makedirs(os.path.dirname(TRANSLATED), exist_ok=True)
    info = json.load(open(INFO)) if os.path.exists(INFO) else {}
    for lang, split in todo:
        t0 = time.time()
        en = pd.read_csv(ENG.format(split), dtype=str, keep_default_na=False)
        out = translate_df(en, lang, tr, model_name)
        empty = out.comment.str.strip() == ""
        assert not empty.any(), f"{int(empty.sum())} empty translations in {lang} {split}"
        out.to_csv(TRANSLATED.format(lang, split), index=False)
        info[f"{lang}_{split}"] = {"model": model_name, "rows": len(out), "seconds": round(time.time() - t0, 1),
                                   "script_share_native": round(float((out.comment.map(script_id) == lang).mean()), 3)}
        print(f"  {lang} {split}: {len(out)} rows in {time.time() - t0:.0f}s, e.g. {out.comment.iloc[0][:80]}")
        if split == "test" and not os.path.exists(REVIEW.format(lang)):
            os.makedirs(os.path.dirname(REVIEW), exist_ok=True)
            review_sample(en, out).to_csv(REVIEW.format(lang), index=False)
    json.dump(info, open(INFO, "w"), indent=1)


def selfcheck():
    en = pd.DataFrame({"id": [f"e{i}" for i in range(120)], "comment": [f"sentence {'x' * (i % 7)} {i}" for i in range(120)],
                       "stance": (LABELS * 40), "split": "test", "source": "GWSD"})
    fake = lambda texts, tgt: [f"<{tgt}> {t}" for t in texts]  # noqa: E731
    out = translate_df(en, "bn", fake, "fake")
    assert out.source_id.tolist() == en.id.tolist() and out.id.iloc[0] == "bn_e0"  # order preserved despite sorting
    assert out.comment.iloc[5] == "<ben_Beng> " + clean(en.comment.iloc[5])
    assert out.stance.tolist() == en.stance.tolist() and out.source.iloc[0] == "machine-translated from GWSD"
    rv = review_sample(en, out)
    assert len(rv) == 100 and rv.stance.value_counts().min() >= 33 and (rv.english != "").all()
    print("translate.py self-check passed (no model download)")


if __name__ == "__main__":
    selfcheck()
    if "--check" not in sys.argv:
        main()
