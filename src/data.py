"""Schema-tolerant loader + validation gate. Self-check: python src/data.py"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import script_id  # noqa: E402

LABELS = ["Favour", "Against", "None"]  # fixed order = label ids 0,1,2 everywhere
CLAIM = "Climate change and global warming is a serious concern."

# spelling variants -> canonical label. Numeric / 'positive' / 'negative' are NOT guessed: pass label_map= for those.
CANON = {
    "favour": "Favour", "favor": "Favour", "favourable": "Favour", "support": "Favour", "supports": "Favour", "agree": "Favour",
    "against": "Against", "oppose": "Against", "opposes": "Against", "deny": "Against", "denial": "Against", "disagree": "Against",
    "none": "None", "neutral": "None", "no stance": "None", "unrelated": "None",
}
# column-name candidates, in priority order, compared case-insensitively
TEXT_COLS = ["comment", "text", "tweet", "sentence", "content", "body", "post", "message"]
LABEL_COLS = ["stance", "label", "labels", "class", "predicted label"]
ID_COLS = ["id", "comment_id", "tweet_id", "index"]


def _sniff(columns, candidates, what):
    lower = {c.strip().lower(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    raise ValueError(f"cannot find a {what} column among {list(columns)}; tried {candidates}")


def load(path, need_label=True, default_source=None, on_blank="raise", label_map=None):
    """Return DataFrame[id, comment, stance, language, source, ...]. Raises on unmapped labels, never silently drops.

    on_blank='drop' removes rows whose label cell is empty (e.g. annotator ties) and keeps them in df.attrs['dropped_blank'].
    """
    # gotcha 1: the literal string "None" is a class label; default read_csv turns it into NaN and deletes a class
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = pd.DataFrame(index=raw.index)
    text_col = _sniff(raw.columns, TEXT_COLS, "text")
    df["comment"] = raw[text_col]
    try:
        df["id"] = raw[_sniff(raw.columns, ID_COLS, "id")]
    except ValueError:
        df["id"] = [f"row{i}" for i in range(len(raw))]  # unlabeled/test files may have no id column
    extra = [c for c in raw.columns if c not in (text_col,) and c.strip().lower() not in LABEL_COLS + ID_COLS]
    for c in extra:
        df[c] = raw[c]

    if need_label:
        label_col = _sniff(raw.columns, LABEL_COLS, "label")
        mapping = {**CANON, **{k.lower(): v for k, v in (label_map or {}).items()}}
        stripped = raw[label_col].str.strip()
        blank = stripped == ""
        if blank.any() and on_blank == "raise":
            raise ValueError(f"{path}: {blank.sum()} rows have an empty label (pass on_blank='drop' to remove them explicitly)")
        mapped = stripped.str.lower().map(mapping)
        bad = sorted(set(stripped[~blank & mapped.isna()]))
        if bad:
            raise ValueError(f"{path}: unmapped label values {bad}; known: {sorted(mapping)}")
        df["stance"] = mapped
        if blank.any():
            df.attrs["dropped_blank"] = df[blank].drop(columns="stance")
            df = df[~blank]

    # 'agreement' mixes numbers (GWStance, 8-annotator vote) with the string 'gold' (SemEval published label)
    if "agreement" in df.columns:
        gold = df["agreement"].str.strip().str.lower() == "gold"
        df["label_source"] = np.where(gold, "published_gold", "majority_vote_8")
        df["agreement"] = pd.to_numeric(df["agreement"].where(~gold, ""), errors="coerce")
    if "source" not in df.columns:
        df["source"] = default_source or os.path.basename(path)
    if "language" not in df.columns:
        df["language"] = df["comment"].map(script_id)  # script-based guess, not a language-ID model
    return df.reset_index(drop=True)


def validate(df, name="dataset", labelled=True):
    """Fail loudly before training on a broken file. Prints the per-source class distribution."""
    problems = []
    if labelled:
        if df["stance"].isna().any() or (df["stance"].astype(str).str.strip() == "").any():
            problems.append("empty/NaN labels")
        extra = set(df["stance"].dropna()) - set(LABELS)
        if extra:
            problems.append(f"labels outside {LABELS}: {sorted(extra)}")
    if (df["comment"].astype(str).str.strip() == "").any():
        problems.append("empty comments")
    dups = int(df["comment"].duplicated().sum())
    if dups:
        problems.append(f"{dups} duplicate comments")
    if (df["source"].astype(str).str.strip() == "").any() or df["source"].isna().any():
        problems.append("rows with empty source")
    if problems:
        raise ValueError(f"VALIDATION FAILED for {name}: " + "; ".join(problems))
    if labelled:
        print(f"[validate] {name}: {len(df)} rows OK; class distribution per source:")
        print(pd.crosstab(df["source"], df["stance"]).reindex(columns=LABELS, fill_value=0).to_string())
    return df


def load_english(path="data/english_stance_merged.csv", ties_out="data/gwstance_ambiguous_ties.csv"):
    """User-supplied GWStance+SemEval file. Blank-stance rows are annotator ties: written to ties_out and dropped."""
    df = load(path, on_blank="drop")
    ties = df.attrs.get("dropped_blank")
    if ties is not None and len(ties):
        ties.to_csv(ties_out, index=False)
        print(f"[data] {len(ties)} tie rows (no annotator majority) dropped -> {ties_out}; {len(df)} labelled rows kept")
    return validate(df, os.path.basename(path))


if __name__ == "__main__":
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.csv")
    pd.DataFrame({"Comment": ["a", "b", "c", "d"], "Label": ["None", "Favor", "AGAINST", "neutral"]}).to_csv(p, index=False)
    # gotcha 1 regression: naive pandas loses the class, ours keeps it
    assert pd.read_csv(p)["Label"].isna().sum() == 1
    out = load(p, default_source="unit-test")
    assert out["stance"].tolist() == ["None", "Favour", "Against", "None"]  # naive read would have given NaN, not "None"
    assert (out["stance"] == "None").sum() == 2
    # unmapped label must raise, not drop
    pd.DataFrame({"text": ["a"], "label": ["maybe"]}).to_csv(p, index=False)
    try:
        load(p)
        raise AssertionError("unmapped label did not raise")
    except ValueError as e:
        assert "maybe" in str(e)
    # blank label raises by default, is dropped (and kept aside) only when asked
    pd.DataFrame({"text": ["a", "b"], "stance": ["Favour", ""], "agreement": ["0.875", ""], "id": ["1", "2"]}).to_csv(p, index=False)
    try:
        load(p)
        raise AssertionError("blank label did not raise")
    except ValueError:
        pass
    kept = load(p, on_blank="drop", default_source="t")
    assert len(kept) == 1 and len(kept.attrs["dropped_blank"]) == 1
    # agreement split
    pd.DataFrame({"text": ["a", "b"], "stance": ["Favour", "None"], "agreement": ["0.75", "gold"]}).to_csv(p, index=False)
    ag = load(p, default_source="t")
    assert ag["label_source"].tolist() == ["majority_vote_8", "published_gold"]
    assert ag["agreement"].iloc[0] == 0.75 and np.isnan(ag["agreement"].iloc[1])
    # validation gate
    ok = pd.DataFrame({"comment": ["x", "y"], "stance": ["Favour", "None"], "source": ["s", "s"]})
    validate(ok, "ok")
    for bad, why in [(ok.assign(stance=["Favour", "Maybe"]), "label"), (ok.assign(comment=["x", "x"]), "dup"),
                     (ok.assign(source=["s", ""]), "source"), (ok.assign(stance=["Favour", np.nan]), "nan")]:
        try:
            validate(bad, why)
            raise AssertionError(f"validate accepted bad file: {why}")
        except ValueError:
            pass
    print("data.py self-check passed")
