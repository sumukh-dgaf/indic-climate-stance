"""Rebuild the English training files from their public sources (needs internet).

  python src/build_english_merged.py            # writes data/english_stance_merged.csv, _highconf.csv, gwstance_ambiguous_ties.csv
  python src/build_english_merged.py --check    # builds in memory and compares with the files already in data/
  python src/build_english_merged.py --force    # overwrite existing files (default: refuse)

Rules (these are the rules the team's supplied files follow; --check verifies it):
  1. GWSD (Luo et al. 2020): each sentence has 8 annotator labels (agrees / disagrees / neutral).
     stance = the unique most frequent label (agrees->Favour, disagrees->Against, neutral->None).
     agreement = count of that label / 8. If two labels share the top count (a tie) stance is left BLANK.
  2. SemEval-2016 Task 6 climate target via TweetEval: published gold label, agreement = 'gold'; the '#SemST' tag is removed.
  3. Concatenate GWSD then SemEval and drop rows with duplicate comment text (first kept).
  4. merged keeps the blank-stance tie rows; src/data.py drops them (and writes the ties file) when loading.
  5. highconf = GWSD rows with agreement >= 0.75 (ties excluded) + all SemEval rows.
"""
import os
import sys
import urllib.request
from collections import Counter

import pandas as pd

GWSD_URL = "https://raw.githubusercontent.com/yiweiluo/GWStance/HEAD/GWSD.tsv"
WORKERS = [f"worker_{i}" for i in range(8)]
VOTE = {"agrees": "Favour", "disagrees": "Against", "neutral": "None"}
SRC_GW = "GWStance/GWSD (Luo et al. 2020), majority vote of 8 annotators"
SRC_SE = "SemEval-2016 Task 6 (via TweetEval)"
COLS = ["id", "comment", "stance", "language", "target", "split", "source", "agreement"]


def majority(row):
    ranked = Counter(VOTE[row[w]] for w in WORKERS).most_common()
    top = ranked[0][1]
    winners = [lab for lab, n in ranked if n == top]
    return pd.Series({"stance": winners[0] if len(winners) == 1 else "", "agreement": top / 8})


def gwsd(cache="data/raw/GWSD.tsv"):
    if not os.path.exists(cache):  # the repository has no LICENSE: fetch from source, never redistribute
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        open(cache, "wb").write(urllib.request.urlopen(GWSD_URL, timeout=120).read())
    raw = pd.read_csv(cache, sep=chr(9), dtype=str, keep_default_na=False)  # tab-separated
    assert len(raw) == 2300, len(raw)  # published size of GWSD.tsv
    v = raw.apply(majority, axis=1)
    return pd.DataFrame({"id": "gwstance_" + raw.guid, "comment": raw.sentence.str.strip(), "stance": v.stance, "language": "en",
                         "target": "global warming is a serious concern",
                         "split": raw.in_held_out_test.map({"True": "test", "False": "train"}),
                         "source": SRC_GW, "agreement": v.agreement.map(str)})


def semeval():
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "stance_climate")
    names = ds["train"].features["label"].names
    m = {"none": "None", "against": "Against", "favor": "Favour"}
    rows = []
    for split, tag in [("train", "train"), ("validation", "val"), ("test", "test")]:
        for i, r in enumerate(ds[split]):
            rows.append({"id": f"climate_{tag}_{i}", "comment": r["text"].replace("#SemST", "").strip(),
                         "stance": m[names[r["label"]]], "language": "en", "target": "Climate Change is a Real Concern",
                         "split": tag, "source": SRC_SE, "agreement": "gold"})
    return pd.DataFrame(rows)


def build():
    both = pd.concat([gwsd(), semeval()], ignore_index=True)[COLS]
    merged = both.drop_duplicates("comment", keep="first").reset_index(drop=True)
    num = pd.to_numeric(merged.agreement, errors="coerce")
    is_gw = merged.source == SRC_GW
    high = merged[(~is_gw) | ((num >= 0.75) & (merged.stance != ""))].reset_index(drop=True)
    return merged, high


if __name__ == "__main__":
    merged, high = build()
    print(f"merged {len(merged)} rows ({(merged.stance == '').sum()} blank-stance ties), highconf {len(high)} rows")
    if "--check" in sys.argv:
        for name, df in [("english_stance_merged.csv", merged), ("english_stance_highconf.csv", high)]:
            old = pd.read_csv("data/" + name, dtype=str, keep_default_na=False)
            same_ids = set(old.id) == set(df.id)
            j = old.merge(df, on="id", suffixes=("_old", "_new"))
            diffs = {c: int((j[c + "_old"] != j[c + "_new"]).sum()) for c in COLS if c != "id"}
            print(f"{name}: ids identical={same_ids}  rows old/new={len(old)}/{len(df)}  field differences={diffs}")
    else:
        targets = {"data/english_stance_merged.csv": merged, "data/english_stance_highconf.csv": high,
                   "data/gwstance_ambiguous_ties.csv": merged[merged.stance == ""]}
        exists = [p for p in targets if os.path.exists(p)]
        if exists and "--force" not in sys.argv:
            sys.exit(f"refusing to overwrite {exists}; use --check to compare or --force to replace")
        for path, df in targets.items():
            df.to_csv(path, index=False, lineterminator="\n")
            print(f"wrote {path} ({len(df)} rows)")
