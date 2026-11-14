"""Fixed English train/val/test splits -> data/splits/english_{train,val,test}.csv (git-ignored: GWStance has no licence).

Test = the official held-out portions already marked in the `split` column (GWSD in_held_out_test + SemEval test).
The rest (including SemEval's 40-row val) is split 85/15 train/val, stratified by label AND source, seed 42.
Run: python src/make_splits.py        Self-check only: python src/make_splits.py --check
"""
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import LABELS, load_english  # noqa: E402

OUT = "data/splits"
EXPECTED = {"train": 1750, "val": 309, "test": 349}


def make_splits(df, seed=42):
    test = df[df.split == "test"]
    rest = df[df.split != "test"]
    tr, va = train_test_split(rest, test_size=0.15, stratify=rest.stance + "|" + rest.source, random_state=seed)
    return {"train": tr.assign(split="train"), "val": va.assign(split="val"), "test": test.assign(split="test")}


def check(splits, expected=EXPECTED):
    for name, d in splits.items():
        assert len(d) == expected[name], f"{name}: {len(d)} rows, expected {expected[name]}"
        assert d.stance.isin(LABELS).all() and (d.stance.str.strip() != "").all(), f"{name}: empty/unknown labels"
    ids = [set(d.id) for d in splits.values()]
    texts = [set(d.comment) for d in splits.values()]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not ids[i] & ids[j] and not texts[i] & texts[j], "overlap between splits"


def main():
    df = load_english()
    splits = make_splits(df)
    check(splits)
    os.makedirs(OUT, exist_ok=True)
    for name, d in splits.items():
        d.to_csv(f"{OUT}/english_{name}.csv", index=False)
        print(f"{name}: {len(d)} rows ", d.stance.value_counts().reindex(LABELS).to_dict())
    # read back with keep_default_na=False: the label "None" must survive the round trip
    back = pd.read_csv(f"{OUT}/english_test.csv", dtype=str, keep_default_na=False)
    assert (back.stance == "None").sum() == (splits["test"].stance == "None").sum()
    print(f"wrote {OUT}/english_{{train,val,test}}.csv")


def selfcheck():
    toy = pd.DataFrame({"id": [str(i) for i in range(60)], "comment": [f"c{i}" for i in range(60)],
                        "stance": LABELS * 20, "source": ["s1", "s2"] * 30, "split": ["test"] * 12 + ["train"] * 48})
    s = make_splits(toy)
    check(s, {"train": 40, "val": 8, "test": 12})  # 48 * 0.15 = 7.2 -> sklearn rounds the test side up to 8
    try:
        check(s, {"train": 41, "val": 7, "test": 12})
        raise AssertionError("size check did not fire")
    except AssertionError as e:
        assert "expected" in str(e)
    print("make_splits.py self-check passed")


if __name__ == "__main__":
    selfcheck()
    if "--check" not in sys.argv:
        main()
