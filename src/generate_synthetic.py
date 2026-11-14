"""Synthetic Hindi/Bengali stance comments with the Gemini API -> data/synthetic/indic_synthetic_v1.csv.

These are SYNTHETIC (LLM-generated) comments, not real YouTube comments. Pipeline (every step resumable):
  1. generate  small JSON batches (lang, label, phenomenon in the prompt), appended to data/synthetic/raw/generate.jsonl
  2. filter    format/length -> script check -> exact duplicates -> near-duplicates (char TF-IDF cosine > 0.85)
  3. judge     a DIFFERENT Gemini model sees only the claim + comment and predicts the stance; keep agreements
  4. select    down to the quota table, split test/val/train, drop train rows near-duplicating val/test
Run:            python src/generate_synthetic.py --run         (needs GEMINI_API_KEY in the environment or .env)
Rebuild only:   python src/generate_synthetic.py --no-api      (filters + splits from the raw files already on disk)
Self-check:     python src/generate_synthetic.py               (no API calls)
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CLAIM, LABELS  # noqa: E402
from preprocess import clean, script_id  # noqa: E402

# Free-tier text models per Google's pricing page (checked 2026-11-14). Google no longer publishes per-model free
# RPM/RPD numbers, so the code backs off on HTTP 429 and stops cleanly when a per-day quota is hit.
GEN_MODEL = "gemini-3.5-flash-lite"
JUDGE_MODEL = "gemini-3.1-flash-lite"  # a different model, so the judge is not grading its own output
API = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
RAW = "data/synthetic/raw"
OUT = "data/synthetic/indic_synthetic_v1.csv"
STATS = "data/synthetic/generation_stats.json"
REVIEW = "data/review/synthetic_test_review.csv"
SOURCE = "synthetic (LLM-generated)"
BATCH, JUDGE_BATCH, OVERGEN, MIN_GAP = 20, 25, 1.25, 4.5  # MIN_GAP seconds between calls to one model (~13 RPM)
NEAR_DUP = 0.85
LEAK = 0.75  # train vs val/test uses a stricter threshold: global dedup already removed everything above 0.85
SEED = 42

# Kept rows per (language, label). Roughly 25% plain, 15% codemix, 15% roman; each column sums to 500.
STANCE_PHEN = {"plain": 125, "codemix": 75, "roman": 75, "negation": 60, "sarcasm": 60, "implicit": 60, "emoji": 45}
NONE_PHEN = {"plain": 125, "codemix": 75, "roman": 75, "offtopic": 70, "question": 55, "neutral_fact": 60, "emoji": 40}
QUOTA = {(lang, lab, ph): n for lang in ("hi", "bn") for lab in LABELS
         for ph, n in (NONE_PHEN if lab == "None" else STANCE_PHEN).items()}
SPLIT_N = {"test": 100, "val": 50}  # per (language, label); train gets the rest

LANG = {"hi": "Hindi", "bn": "Bengali"}
SCRIPT = {"hi": "Devanagari", "bn": "Bengali"}
LABEL_DEF = {
    "Favour": "agrees that climate change / global warming is a serious concern (worry, urgency, blaming pollution, "
              "asking for action, praising climate action)",
    "Against": "disagrees: denies, minimises or mocks the concern, with realistic denial arguments such as 'it was always "
               "this hot', 'a Western conspiracy to stop India developing', 'scientists just want funding', 'it snowed / "
               "this winter was freezing so where is the warming', 'media hype', 'nature/God decides the weather'",
    "None": "takes NO stance on whether climate change is a serious concern (unrelated, neutral or unclear)",
}
PHEN_DEF = {
    "plain": "written fully in {script} script",
    "codemix": "{script} script mixed with English words typed in Latin letters, as in real Indian YouTube comments",
    "roman": "{lang} typed entirely in Latin letters (Romanised, phone-keyboard style), NOT English",
    "negation": "in {script} script; the stance is carried by a negation ('it is not true that...', 'nobody can deny...')",
    "sarcasm": "in {script} script; sarcastic or ironic, the literal words say the opposite of what is meant",
    "implicit": "in {script} script; the stance is clear but there is NO climate keyword (no 'climate', 'global "
                "warming', 'जलवायु', 'জলবায়ু'): talk about heat, rain, rivers, crops, AC, smoke instead",
    "emoji": "in {script} script with 1-4 emoji that carry part of the meaning",
    "offtopic": "in {script} script; about something unrelated to climate (the anchor, audio quality, cricket, a song)",
    "question": "in {script} script; a genuine question that reveals no opinion",
    "neutral_fact": "in {script} script; a neutral fact or news-like statement about weather/climate, no opinion",
}
NONE_TRAPS = ("Include typical YouTube noise in some comments: 'nice video', 'link please', a timestamp like 3:45, "
              "'first comment', 'who is watching in 2026', asking for the background song name.")
TOPICS = ["heatwaves", "floods", "cyclones", "Himalayan glaciers", "crop loss", "cold winters", "coal power",
          "electric vehicles", "government policy", "climate summits", "media hype", "scientists", "air pollution",
          "monsoon delays", "water shortage", "rising electricity bills"]
TONES = ["angry", "worried", "mocking", "indifferent", "devotional", "joking", "tired", "hopeful"]


class QuotaExhausted(Exception):
    pass


# ---------------- API
_last_call = {}


def api_key():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key and os.path.exists(".env"):  # .env is git-ignored; KEY=value lines
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def call(model, prompt, key, temperature):
    """One generateContent call returning parsed JSON, or None on an unusable answer. Never logs the key."""
    import requests
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}}
    for attempt in range(6):
        gap = MIN_GAP - (time.time() - _last_call.get(model, 0))
        if gap > 0:
            time.sleep(gap)
        _last_call[model] = time.time()
        try:
            # key in a header, not the URL, so it cannot appear in an error message or log
            r = requests.post(API.format(model), headers={"x-goog-api-key": key}, json=body, timeout=180)
        except requests.RequestException as e:
            print(f"  network error ({type(e).__name__}), retrying")
            time.sleep(20)
            continue
        if r.status_code == 200:
            return parse_json(r.json())
        if r.status_code == 429 and "PerDay" in r.text:
            raise QuotaExhausted(f"{model}: daily free quota used up")
        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(300, 30 * 2 ** attempt)
            print(f"  HTTP {r.status_code} from {model}, backing off {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"{model}: HTTP {r.status_code}: {r.text[:300]}")
    raise QuotaExhausted(f"{model}: still rate-limited after 6 retries")


def parse_json(resp):
    try:
        parts = resp["candidates"][0]["content"]["parts"]
        out = json.loads([p["text"] for p in parts if "text" in p][-1])
    except (KeyError, IndexError, json.JSONDecodeError):
        return None  # blocked or malformed answer: the batch stays undone and is retried on the next run
    return out if isinstance(out, list) else None


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------- 1. generation
def gen_prompt(batch_key, n=BATCH):
    lang, label, phen, _ = batch_key.split("|")
    rng = random.Random(batch_key)  # topics/tones vary per batch but are reproducible
    style = PHEN_DEF[phen].format(script=SCRIPT[lang], lang=LANG[lang])
    return (f"You are helping build a research dataset for stance detection on Indian social media.\n"
            f"Write {n} different, realistic YouTube comments in {LANG[lang]}, posted under videos about weather and climate.\n"
            f'Claim: "{CLAIM}"\n'
            f"Every comment must have the stance {label.upper()} toward the claim: the commenter {LABEL_DEF[label]}.\n"
            f"Style of every comment: {style}.\n"
            f"Mix these topics: {', '.join(rng.sample(TOPICS, 4))}. Mix these tones: {', '.join(rng.sample(TONES, 3))}.\n"
            "Rules: 3 to 40 words each, varied lengths; informal, with natural spelling variation and small typos; "
            "every comment worded and opened differently; no real usernames, no personal details, no hate speech.\n"
            + (NONE_TRAPS + "\n" if label == "None" else "")
            + f"Return only a JSON array of {n} strings.")


def planned_batches(quota):
    return [f"{lang}|{lab}|{ph}|{i}" for (lang, lab, ph), q in quota.items() for i in range(math.ceil(q * OVERGEN / BATCH))]


def generate(batches, key):
    path = f"{RAW}/generate.jsonl"
    done = {r["batch"] for r in read_jsonl(path)}
    todo = [b for b in batches if b not in done]
    print(f"[generate] {len(done)} batches on disk, {len(todo)} to call ({GEN_MODEL})")
    for k, b in enumerate(todo, 1):
        items = call(GEN_MODEL, gen_prompt(b), key, temperature=1.0)
        if items is None:
            print(f"  {b}: unusable answer, will retry on next run")
            continue
        append_jsonl(path, {"batch": b, "model": GEN_MODEL, "items": items, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        if k % 10 == 0:
            print(f"  {k}/{len(todo)} batches")


def raw_rows():
    rows = []
    for r in read_jsonl(f"{RAW}/generate.jsonl"):
        lang, label, phen, _ = r["batch"].split("|")
        for it in r["items"]:
            rows.append({"comment": it.strip() if isinstance(it, str) else None, "stance": label, "language": lang,
                         "phenomenon": phen, "generator_model": r["model"]})
    return pd.DataFrame(rows, columns=["comment", "stance", "language", "phenomenon", "generator_model"])


# ---------------- 2. filters
def script_ok(text, lang, phen):
    """A Hindi row must be Devanagari, mixed, or Latin as its phenomenon requires (same for Bengali)."""
    words = {script_id(w) for w in text.split()}
    if ("bn" if lang == "hi" else "hi") in words:
        return False  # the other Indic script
    if phen == "roman":
        return script_id(text) == "latin"
    if phen == "codemix":
        return lang in words and "latin" in words
    return script_id(text) == lang


def length_ok(text):
    return isinstance(text, str) and 3 <= len(text.split()) <= 40


def norm(text):
    return clean(text).lower()


def near_dup_mask(texts, against=None, thr=NEAR_DUP):
    """Greedy: True = keep. Without `against`, drops a text too similar to an earlier kept one; with it, drops texts
    too similar to any text in `against`. Character 2-4 gram TF-IDF cosine."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [norm(t) for t in texts]
    ref = [norm(t) for t in against] if against is not None else []
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(texts + ref)
    A = X.transform(texts)
    if against is not None:
        return np.asarray((A @ X.transform(ref).T).max(axis=1).todense()).ravel() <= thr
    S = (A @ A.T).toarray()
    keep = np.ones(len(texts), dtype=bool)
    for i in range(len(texts)):
        if (S[i, :i][keep[:i]] > thr).any():
            keep[i] = False
    return keep


def tag(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ---------------- 3. judge
def judge_prompt(comments):
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(comments))
    return (f'Stance detection. Claim: "{CLAIM}"\n'
            "For each comment (Hindi or Bengali, possibly Romanised or mixed with English), give the commenter's stance "
            "toward the claim: Favour = agrees it is a serious concern; Against = denies, minimises or mocks it; "
            "None = no stance, unrelated or neutral. For sarcasm, give the stance actually meant.\n"
            f"Comments:\n{numbered}\n"
            f'Return only a JSON array of {len(comments)} strings, each exactly "Favour", "Against" or "None", in order.')


def judged():
    return {h: p for r in read_jsonl(f"{RAW}/judge.jsonl") for h, p in zip(r["hashes"], r["preds"])}


def judge(texts, key):
    path = f"{RAW}/judge.jsonl"
    have = judged()
    todo = [t for t in dict.fromkeys(texts) if tag(t) not in have]
    random.Random(SEED).shuffle(todo)  # mix labels within a batch so the batch itself does not hint at the label
    print(f"[judge] {len(have)} comments judged on disk, {len(todo)} to call ({JUDGE_MODEL})")
    for i in range(0, len(todo), JUDGE_BATCH):
        chunk = todo[i:i + JUDGE_BATCH]
        preds = call(JUDGE_MODEL, judge_prompt(chunk), key, temperature=0.0)
        if preds is None or len(preds) != len(chunk) or not set(preds) <= set(LABELS):
            print("  judge answer unusable, will retry on next run")
            continue
        append_jsonl(path, {"model": JUDGE_MODEL, "hashes": [tag(t) for t in chunk], "preds": preds})


# ---------------- pipeline
def prefilter(df):
    """Step 2, the API-free filters. Returns survivors (the only rows worth judging) and a count per filter."""
    n = {"generated": len(df)}
    df = df[df.comment.map(length_ok)]
    n["removed_format_or_length"] = n["generated"] - len(df)
    ok = [script_ok(t, l, p) for t, l, p in zip(df.comment, df.language, df.phenomenon)]
    n["removed_script"] = len(df) - sum(ok)
    df = df[ok]
    before = len(df)
    df = df[~df.comment.map(norm).duplicated()]
    n["removed_exact_duplicate"] = before - len(df)
    keep = near_dup_mask(df.comment.tolist()) if len(df) else np.array([], bool)
    n["removed_near_duplicate"] = int((~keep).sum())
    return df[keep].copy(), n


def apply_filters(df):
    """Steps 2 and 3 (judge verdicts read from disk). Returns survivors and a count for every filter."""
    df, n = prefilter(df)
    verdict = judged()
    df["judge"] = df.comment.map(lambda t: verdict.get(tag(t), ""))
    n["awaiting_judge"] = int((df.judge == "").sum())
    jd = df[df.judge != ""]
    n["judged"] = len(jd)
    n["removed_judge_disagrees"] = int((jd.judge != jd.stance).sum())
    rate = lambda g: {k: round(float(v), 3) for k, v in g.apply(lambda x: (x.judge == x.stance).mean()).items()}  # noqa: E731
    n["judge_agreement_overall"] = round(float((jd.judge == jd.stance).mean()), 3) if len(jd) else None
    if len(jd):
        n["judge_agreement_by_label"] = rate(jd.groupby("stance"))
        n["judge_agreement_by_phenomenon"] = rate(jd.groupby("phenomenon"))
        n["judge_agreement_by_language"] = rate(jd.groupby("language"))
        n["judge_confusion_intended_vs_judge"] = pd.crosstab(jd.stance, jd.judge).to_dict()
    return jd[jd.judge == jd.stance].drop(columns="judge"), n


def select_and_split(df, quota=QUOTA, split_n=SPLIT_N):
    from sklearn.model_selection import train_test_split
    picked, short = [], {}
    for (lang, lab, ph), q in quota.items():
        g = df[(df.language == lang) & (df.stance == lab) & (df.phenomenon == ph)]
        if len(g) < q:
            short["|".join((lang, lab, ph))] = q - len(g)
        picked.append(g.sample(min(q, len(g)), random_state=SEED))
    df = pd.concat(picked).sort_values(["language", "stance", "phenomenon"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby(["language", "stance"]):
        rest = g
        for split, k in split_n.items():
            strat = rest.phenomenon if rest.phenomenon.value_counts().min() >= 2 else None
            rest, take = train_test_split(rest, test_size=min(k, len(rest) - 1), stratify=strat, random_state=SEED)
            parts.append(take.assign(split=split))
        parts.append(rest.assign(split="train"))
    df = pd.concat(parts)
    ev = df[df.split != "train"]
    leak = ~near_dup_mask(df[df.split == "train"].comment.tolist(), against=ev.comment.tolist(), thr=LEAK)
    leak_ids = df[df.split == "train"].index[leak]
    df = df.drop(index=leak_ids).sort_values(["split", "language", "stance", "phenomenon"]).reset_index(drop=True)
    df.insert(0, "id", [f"syn_{l}_{i:05d}" for i, l in enumerate(df.language)])
    df["source"] = SOURCE
    df["human_verified"] = "no"
    return df, short, int(leak.sum())


def build(quota=QUOTA):
    raw = raw_rows()
    if raw.empty:
        print("no generated batches on disk yet")
        return None
    kept, n = apply_filters(raw)
    final, short, n_leak = select_and_split(kept, quota)
    n.update({"passed_all_filters": len(kept), "kept_after_quota": len(final), "removed_train_leak_vs_eval": n_leak,
              "shortfall_by_cell": short, "split_sizes": final.split.value_counts().to_dict(),
              "by_language_label": {f"{l}|{s}": int(v) for (l, s), v in final.groupby(["language", "stance"]).size().items()},
              "by_phenomenon": final.phenomenon.value_counts().to_dict(),
              "generator_model": GEN_MODEL, "judge_model": JUDGE_MODEL,
              "note": "synthetic (LLM-generated) comments; test labels not yet human-verified until apply_review.py runs"})
    cols = ["id", "comment", "stance", "language", "phenomenon", "split", "source", "generator_model", "human_verified"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    final[cols].to_csv(OUT, index=False)
    json.dump(n, open(STATS, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    write_review(final)
    print(json.dumps({k: v for k, v in n.items() if not k.startswith("judge_confusion")}, indent=1, ensure_ascii=False))
    print(f"wrote {OUT} ({len(final)} rows), {STATS}, {REVIEW}")
    return n


def write_review(final):
    # blind review: reviewers do NOT see the intended label, so they are not anchored by it (apply_review.py joins by id)
    test = final[final.split == "test"].sample(frac=1, random_state=SEED)
    os.makedirs(os.path.dirname(REVIEW), exist_ok=True)
    if os.path.exists(REVIEW) and (pd.read_csv(REVIEW, dtype=str, keep_default_na=False).reviewer_label != "").any():
        print(f"{REVIEW} already has reviewer labels: not overwritten")
        return
    test.assign(reviewer="", reviewer_label="", comment_is_natural="")[
        ["id", "language", "phenomenon", "comment", "reviewer", "reviewer_label", "comment_is_natural"]].to_csv(REVIEW, index=False)


def deficits(kept, quota):
    out = {}
    for (lang, lab, ph), q in quota.items():
        have = ((kept.language == lang) & (kept.stance == lab) & (kept.phenomenon == ph)).sum()
        if have < q:
            out[(lang, lab, ph)] = q - have
    return out


def run(topup_rounds=2):
    key = api_key()
    if not key:
        sys.exit("GEMINI_API_KEY is not set (environment or .env). Get a free key at https://aistudio.google.com/apikey, "
                 "then: echo GEMINI_API_KEY=... > .env   (.env is git-ignored)")
    try:
        batches = planned_batches(QUOTA)
        for rnd in range(topup_rounds + 1):
            generate(batches, key)
            raw = raw_rows()
            judge(prefilter(raw)[0].comment.tolist(), key)  # judge only rows that survived the cheap filters
            kept, _ = apply_filters(raw)
            need = deficits(kept, QUOTA)
            if not need:
                break
            # top up only the cells the judge thinned out (sarcasm/implicit lose the most); 1.5x for the expected loss
            have_idx = {}
            for b in batches:
                cell = tuple(b.split("|")[:3])
                have_idx[cell] = max(have_idx.get(cell, -1), int(b.split("|")[3]))
            extra = [f"{l}|{s}|{p}|{have_idx[(l, s, p)] + 1 + i}" for (l, s, p), d in need.items()
                     for i in range(math.ceil(d * 1.5 / BATCH))]
            print(f"[top-up round {rnd + 1}] {sum(need.values())} rows short in {len(need)} cells -> {len(extra)} batches")
            batches += extra
    except QuotaExhausted as e:
        print(f"STOPPED: {e}. Re-run later to resume; building the dataset from what exists now.")
    return build()


def selfcheck():
    # quota arithmetic
    for lang in LANG:
        for lab in LABELS:
            assert sum(q for (l, s, _), q in QUOTA.items() if l == lang and s == lab) == 500
    total = sum(QUOTA.values())
    share = lambda ph: sum(q for (_, _, p), q in QUOTA.items() if p == ph) / total  # noqa: E731
    assert total == 3000 and abs(share("plain") - 0.25) < 0.01 and share("codemix") >= 0.12 and share("roman") >= 0.12
    assert sum(SPLIT_N.values()) * 6 == 900  # 600 test + 300 val; train = 2,100 before leak removal
    assert len(planned_batches(QUOTA)) * BATCH >= total * OVERGEN
    # filters on fixed strings
    assert script_ok("गर्मी हर साल बढ़ रही है भाई", "hi", "plain")
    assert not script_ok("गर्मी हर साल बढ़ रही है भाई", "bn", "plain")
    assert script_ok("garmi har saal badh rahi hai bhai", "hi", "roman")
    assert script_ok("ये climate change वाला drama बंद करो", "hi", "codemix")
    assert not script_ok("ये क्लाइमेट चेंज वाला ड्रामा बंद करो", "hi", "codemix")  # no Latin word: not code-mixed
    assert script_ok("এই গরম আর সহ্য হয় না 😓", "bn", "emoji")
    assert not script_ok("এই গরম आर सहन", "bn", "plain")  # Devanagari word inside a Bengali row
    assert length_ok("bahut garmi hai") and not length_ok("garmi") and not length_ok(None)
    keep = near_dup_mask(["global warming is fake bro", "global warming is fake bro!!", "save trees for our kids"])
    assert keep.tolist() == [True, False, True]
    assert near_dup_mask(["save trees for our kids"], against=["save trees for our kids!"], thr=LEAK).tolist() == [False]
    # JSON parsing of a canned API answer
    resp = {"candidates": [{"content": {"parts": [{"text": '["a b c", "d e f"]'}]}}]}
    assert parse_json(resp) == ["a b c", "d e f"] and parse_json({"candidates": []}) is None
    assert "SARCASTIC" not in gen_prompt("hi|Against|sarcasm|0") and "AGAINST" in gen_prompt("hi|Against|sarcasm|0")
    assert gen_prompt("bn|None|offtopic|3") == gen_prompt("bn|None|offtopic|3")  # reproducible prompts
    # select_and_split on a toy pool: exact split sizes, no train/eval leak
    toy_q = {(l, s, p): 6 for l in LANG for s in LABELS for p in ("plain", "roman")}
    pool = pd.DataFrame([{"comment": f"{l} {s} {p} unique comment number {i} " + "xyz"[i % 3] * (i + 1), "stance": s,
                          "language": l, "phenomenon": p, "generator_model": "m"}
                         for l in LANG for s in LABELS for p in ("plain", "roman") for i in range(8)])
    out, short, _ = select_and_split(pool, toy_q, {"test": 4, "val": 2})
    assert not short and (out[out.split == "test"].groupby(["language", "stance"]).size() == 4).all()
    assert out.id.is_unique and set(out.human_verified) == {"no"} and set(out.source) == {SOURCE}
    print("generate_synthetic.py self-check passed (no API calls)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="call the API (resumable), then build the dataset")
    ap.add_argument("--no-api", action="store_true", help="build the dataset from raw files already on disk")
    a = ap.parse_args()
    selfcheck()
    if a.run:
        run()
    elif a.no_api:
        build()
