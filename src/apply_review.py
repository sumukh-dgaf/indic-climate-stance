"""Apply the team's blind review of the synthetic test set to data/synthetic/indic_synthetic_v1.csv.

Reviewers fill data/review/synthetic_test_review.csv: reviewer (name), reviewer_label (Favour/Against/None),
comment_is_natural (yes/no). They never see the intended label. This script reports human-vs-intended agreement,
replaces the label with the human one where they differ, and marks reviewed rows human_verified=yes.
Run: python src/apply_review.py          Self-check only: python src/apply_review.py --check
Progress only (changes nothing): python src/apply_review.py --status
Translation check: python src/apply_review.py --translation  (counts from data/review/translation_review_{hi,bn}.csv)
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CANON, LABELS, SYNTHETIC  # noqa: E402

REVIEW = "data/review/synthetic_test_review.csv"
STATS = "data/review/review_stats.json"


def apply(syn, rev):
    rev = rev[rev.reviewer_label.str.strip() != ""].copy()
    rev["reviewer_label"] = rev.reviewer_label.str.strip().str.lower().map(CANON)
    assert rev.reviewer_label.notna().all(), f"reviewer_label must be one of {LABELS}"
    assert rev.id.isin(syn.id).all(), "review file has ids that are not in the synthetic CSV"
    m = rev.drop(columns=["language", "phenomenon"], errors="ignore").merge(syn[["id", "stance", "language"]], on="id")
    agree = m.reviewer_label == m.stance
    stats = {"reviewed": len(m), "agreement_with_intended": round(float(agree.mean()), 3) if len(m) else None,
             "labels_changed": int((~agree).sum()),
             "agreement_by_language": {k: round(float(v), 3) for k, v in agree.groupby(m.language).mean().items()},
             "agreement_by_intended_label": {k: round(float(v), 3) for k, v in agree.groupby(m.stance).mean().items()},
             "changes_intended_to_human": pd.crosstab(m.stance, m.reviewer_label).to_dict(),
             "comment_is_natural_yes": round(float((m.comment_is_natural.str.strip().str.lower() == "yes").mean()), 3)
             if len(m) else None}
    syn = syn.copy()
    new = dict(zip(rev.id, rev.reviewer_label))
    hit = syn.id.isin(new)
    syn.loc[hit, "stance"] = syn.loc[hit, "id"].map(new)
    syn.loc[hit, "human_verified"] = "yes"
    return syn, stats


def status(rev):
    """Labelled rows per language, so the final run only starts once all 600 are done."""
    done = rev.reviewer_label.str.strip() != ""
    return {lang: f"{int(done[g.index].sum())} of {len(g)}" for lang, g in rev.groupby("language")}, int(done.sum())


def main():
    syn = pd.read_csv(SYNTHETIC, dtype=str, keep_default_na=False)
    rev = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
    syn, stats = apply(syn, rev)
    n_test = (syn.split == "test").sum()
    stats["test_rows_verified"] = f"{(syn[syn.split == 'test'].human_verified == 'yes').sum()} of {n_test}"
    syn.to_csv(SYNTHETIC, index=False)
    json.dump(stats, open(STATS, "w", encoding="utf-8"), indent=1)
    print(json.dumps(stats, indent=1))
    print(f"updated {SYNTHETIC}; stats in {STATS}")


def translation_stats(frames):
    """frames: {lang: filled translation_review df}. Share of reviewed rows whose stance label survives translation."""
    out = {}
    for lang, df in frames.items():
        done = df[df.label_survives.str.strip() != ""]
        yes = done.label_survives.str.strip().str.lower().isin(["yes", "y"])
        out[lang] = {"reviewed": len(done), "label_survives_rate": round(float(yes.mean()), 3) if len(done) else None,
                     "by_stance": {k: round(float(v), 3) for k, v in yes.groupby(done.stance).mean().items()}}
    return out


def main_translation():
    # the review files hold GWStance text (no licence) and stay git-ignored; only these counts are committed
    frames = {l: pd.read_csv(f"data/review/translation_review_{l}.csv", dtype=str, keep_default_na=False)
              for l in ("hi", "bn") if os.path.exists(f"data/review/translation_review_{l}.csv")}
    stats = translation_stats(frames)
    json.dump(stats, open("data/review/translation_review_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1), "\nwrote data/review/translation_review_stats.json")


def selfcheck():
    syn = pd.DataFrame({"id": ["a", "b", "c"], "stance": ["Favour", "None", "Against"], "language": ["hi", "bn", "hi"],
                        "split": "test", "human_verified": "no"})
    rev = pd.DataFrame({"id": ["a", "b", "c"], "reviewer": ["N", "N", ""], "reviewer_label": ["favour", "Against", ""],
                        "comment_is_natural": ["yes", "no", ""]})
    out, st = apply(syn, rev)
    assert st["reviewed"] == 2 and st["labels_changed"] == 1 and st["agreement_with_intended"] == 0.5
    assert out.stance.tolist() == ["Favour", "Against", "Against"]  # b relabelled; c unreviewed and untouched
    assert out.human_verified.tolist() == ["yes", "yes", "no"]
    tr = pd.DataFrame({"stance": ["Favour", "Against", "None"], "label_survives": ["yes", "no", ""]})
    ts = translation_stats({"hi": tr})["hi"]
    assert ts["reviewed"] == 2 and ts["label_survives_rate"] == 0.5
    assert status(rev.assign(language=["hi", "hi", "bn"])) == ({"bn": "0 of 1", "hi": "2 of 2"}, 2)
    print("apply_review.py self-check passed")


if __name__ == "__main__":
    selfcheck()
    if "--status" in sys.argv:  # python src/apply_review.py --status
        by_lang, n = status(pd.read_csv(REVIEW, dtype=str, keep_default_na=False))
        print("labelled:", by_lang, f"| total {n} of 600", "| READY" if n == 600 else "| NOT READY: wait for the team")
    elif "--translation" in sys.argv:  # python src/apply_review.py --translation
        main_translation()
    elif "--check" not in sys.argv:
        main()
