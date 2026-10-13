"""Exploratory data analysis -> outputs/eda_report.md, outputs/eda_stats.json, outputs/fig_*.png

Run: python src/eda.py     (needs data/english_stance_*.csv, data/pilot_indic_stance.csv; external files optional)
"""
import json
import os
import sys
from collections import Counter
from itertools import chain
from math import log

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import LABELS, load, load_english, validate  # noqa: E402
from preprocess import clean, script_id, tokenize  # noqa: E402

OUT = "outputs/"
COLORS = {"Favour": "#2a9d8f", "Against": "#e76f51", "None": "#8d99ae"}
PILOT_NOTE = "team-authored pilot corpus, development only (synthetic, not scraped, not the official data)"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})


def majority_floor(y):
    """Macro-F1 of always predicting the most frequent class: the number any real model must beat."""
    y = list(y)
    top = Counter(y).most_common(1)[0][0]
    return f1_score(y, [top] * len(y), average="macro", labels=LABELS, zero_division=0)


def describe(series):
    return {"mean": round(float(series.mean()), 1), "median": float(series.median()),
            "p95": float(series.quantile(0.95)), "max": float(series.max())}


def stats(df, name):
    cleaned = df.comment.map(clean)
    toks = cleaned.map(tokenize)
    counts = Counter(chain.from_iterable(toks))
    vocab = len(counts)
    hapax = sum(1 for c in counts.values() if c == 1)
    cls = df.stance.value_counts().reindex(LABELS, fill_value=0)
    return {
        "name": name, "n": len(df), "classes": cls.to_dict(),
        "class_pct": (100 * cls / len(df)).round(1).to_dict(),
        "chars": describe(df.comment.str.len()), "words": describe(toks.map(len)),
        "tokens": int(sum(counts.values())), "vocab": vocab, "hapax": hapax,
        "hapax_rate_pct": round(100 * hapax / vocab, 1),
        "duplicates_raw": int(df.comment.duplicated().sum()), "duplicates_clean": int(cleaned.duplicated().sum()),
        "empty_after_clean": int((cleaned == "").sum()), "missing_cells": int(df.isna().sum().sum()),
        "majority_floor_macro_f1": round(float(majority_floor(df.stance)), 4),
    }, toks, counts


def frequent(counts, k=15, drop_english_stop=True):
    items = [(w, c) for w, c in counts.most_common() if len(w) > 1 and not (drop_english_stop and w in ENGLISH_STOP_WORDS)]
    return items[:k]


def discriminative(df, toks, k=8, min_count=5):
    """Smoothed log-odds of a token in one class vs the other two. ponytail: no significance test, ranks only."""
    per = {lab: Counter(chain.from_iterable(toks[df.stance == lab])) for lab in LABELS}
    vocab = set().union(*per.values())
    V = len(vocab)
    out = {}
    for lab in LABELS:
        rest = Counter()
        for o in LABELS:
            if o != lab:
                rest.update(per[o])
        n1, n0 = sum(per[lab].values()), sum(rest.values())
        score = {w: log((c + 1) / (n1 + V)) - log((rest[w] + 1) / (n0 + V))
                 for w, c in per[lab].items() if c + rest[w] >= min_count}
        out[lab] = sorted(score.items(), key=lambda kv: -kv[1])[:k]
    return out


def bar_classes(ax, df_by_name, title):
    names = list(df_by_name)
    x = np.arange(len(names))
    w = 0.26
    for i, lab in enumerate(LABELS):
        pct = [100 * (d.stance == lab).mean() for d in df_by_name.values()]
        cnt = [(d.stance == lab).sum() for d in df_by_name.values()]
        bars = ax.bar(x + (i - 1) * w, pct, w, label=lab, color=COLORS[lab])
        for b, c in zip(bars, cnt):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, str(c), ha="center", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("% of dataset")
    ax.set_title(title, loc="left", fontsize=14)
    ax.legend(frameon=False)


def md_table(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    os.makedirs(OUT, exist_ok=True)
    eng = load_english()
    high = validate(load("data/english_stance_highconf.csv"), "english_stance_highconf.csv")
    pilot = validate(load("data/pilot_indic_stance.csv", default_source="team-authored pilot (synthetic, dev only)"),
                     "pilot_indic_stance.csv")
    ties = pd.read_csv("data/gwstance_ambiguous_ties.csv", dtype=str, keep_default_na=False)

    S, T, C = {}, {}, {}
    for key, df, name in [("english_merged", eng, "English merged (GWSD+SemEval)"),
                          ("english_highconf", high, "English high-confidence"), ("pilot", pilot, "Pilot hi+bn (dev only)")]:
        S[key], T[key], C[key] = stats(df, name)

    # ---- external sources (not training data): size, classes, length only
    ext_rows, ext_stats = [], {}
    for fn in ["semeval2016_climate_en", "dhoroni_bn_news", "multiclimate_en_video", "swami_hinglish_demonetisation"]:
        p = f"data/external/{fn}.csv"
        if not os.path.exists(p):
            continue
        d = load(p)
        words = d.comment.map(lambda t: len(tokenize(clean(t))))
        c = d.stance.value_counts().reindex(LABELS, fill_value=0)
        ext_stats[fn] = {"n": len(d), "classes": c.to_dict(), "words": describe(words),
                         "duplicates": int(d.comment.duplicated().sum()), "target": d.target.iloc[0], "domain": d.domain.iloc[0]}
        ext_rows.append([fn, len(d), f"{c['Favour']}/{c['Against']}/{c['None']}", ext_stats[fn]["words"]["median"],
                         d.target.iloc[0], d.domain.iloc[0]])

    # ---- extra findings
    gw = eng[eng.label_source == "majority_vote_8"]
    n_gw_total = len(gw) + len(ties)
    tie_half = int((ties.agreement == "0.5").sum())
    tie_38 = int((ties.agreement == "0.375").sum())
    pilot_script = pilot.assign(script=pilot.comment.map(script_id)).groupby(["language", "script"]).size().unstack(fill_value=0)
    pilot_phen = pd.crosstab(pilot.phenomenon, pilot.stance).reindex(columns=LABELS)
    # Same vocabulary, different stance: tokens that occur in BOTH Favour and Against English sentences
    shared = set(w for w, c in C["english_merged"].items() if c >= 1)
    fav_v = set(chain.from_iterable(T["english_merged"][eng.stance == "Favour"]))
    ag_v = set(chain.from_iterable(T["english_merged"][eng.stance == "Against"]))
    extra = {"ties": len(ties), "ties_4_4": tie_half, "ties_3_3_2": tie_38, "gwsd_total_with_ties": n_gw_total,
             "ties_pct_of_gwsd": round(100 * len(ties) / n_gw_total, 1),
             "favour_against_vocab_overlap_pct": round(100 * len(fav_v & ag_v) / len(ag_v), 1),
             "shared_vocab_note_vocab_size": len(shared)}
    S["extra"], S["external"] = extra, ext_stats
    json.dump(S, open(OUT + "eda_stats.json", "w"), indent=1)

    # ---- figures
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_classes(ax, {"English merged\n(2,408)": eng, "English high-conf\n(1,686)": high, "Pilot hi+bn\n(150, synthetic)": pilot},
                "Class distribution by dataset (bar label = count)")
    fig.tight_layout()
    fig.savefig(OUT + "fig_class_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    src = {s.split(" (")[0].replace("GWStance/GWSD", "GWSD (news)").replace("SemEval-2016 Task 6", "SemEval (tweets)"): d
           for s, d in eng.groupby("source")}
    bar_classes(ax, src, "English merged: class mix by source")
    fig.tight_layout()
    fig.savefig(OUT + "fig_class_by_source.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_classes(ax, {"Hindi (75)": pilot[pilot.language == "hi"], "Bengali (75)": pilot[pilot.language == "bn"]},
                "Pilot corpus (synthetic): class x language")
    fig.tight_layout()
    fig.savefig(OUT + "fig_pilot_language_class.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, col in [("english_merged", "#264653"), ("pilot", "#e9c46a")]:
        w = T[key].map(len)
        ax.hist(w.clip(upper=60), bins=30, alpha=0.7, density=True, color=col, label=S[key]["name"])
    ax.set_xlabel("words per comment (clipped at 60)")
    ax.set_ylabel("density")
    ax.set_title("Text length", loc="left", fontsize=14)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT + "fig_length_distribution.png", dpi=160)
    plt.close(fig)

    disc = discriminative(eng, T["english_merged"], k=8, min_count=8)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, lab in zip(axes, LABELS):
        words, sc = zip(*disc[lab][::-1])
        ax.barh(words, sc, color=COLORS[lab])
        ax.set_title(f"{lab}-leaning tokens", fontsize=13)
        ax.set_xlabel("log-odds vs other classes")
    fig.tight_layout()
    fig.savefig(OUT + "fig_discriminative_tokens.png", dpi=160)
    plt.close(fig)
    disc_pilot = discriminative(pilot, T["pilot"], k=8, min_count=3)
    top_tok = disc["Favour"][0][0]
    hit = T["english_merged"].map(lambda ts: top_tok in ts)
    artefact = (f"* The top Favour token `{top_tok}` occurs in {int(hit.sum())} rows, "
                f"{100 * (eng[hit].stance == 'Favour').mean():.0f}% of them Favour, nearly all from one source "
                f"({eng[hit].source.str[:8].value_counts().index[0]}): an event-hashtag collection artefact, not a stance cue. "
                "Cross-validation will reward memorising such clusters, so English CV scores are optimistic.")

    # ---- report
    def stat_row(k):
        s = S[k]
        return [s["name"], s["n"], f"{s['classes']['Favour']} / {s['classes']['Against']} / {s['classes']['None']}",
                f"{s['chars']['mean']} / {s['chars']['median']:.0f} / {s['chars']['p95']:.0f} / {s['chars']['max']:.0f}",
                f"{s['words']['mean']} / {s['words']['median']:.0f} / {s['words']['p95']:.0f} / {s['words']['max']:.0f}",
                s["vocab"], f"{s['hapax_rate_pct']}%", s["duplicates_raw"], s["majority_floor_macro_f1"]]

    L = ["# Exploratory data analysis (generated by src/eda.py)", "",
         "> Every number below was computed by `src/eda.py` on the files in this repo's `data/`. "
         f"The pilot rows are a **{PILOT_NOTE}**; they must never be quoted as official-set results.", "",
         "## 1. Datasets analysed", "",
         md_table([stat_row(k) for k in ("english_merged", "english_highconf", "pilot")],
                  ["dataset", "rows", "Favour / Against / None", "chars mean/med/p95/max", "words mean/med/p95/max",
                   "vocab", "hapax rate", "dup. comments", "majority-class macro-F1 floor"]), "",
         "Tokens are lower-cased words after `preprocess.clean`, split with the Indic-aware `TOKEN` regex. "
         "*Hapax rate* = share of vocabulary types that occur exactly once.", "",
         "## 2. External sources (not part of the training pool)", "",
         md_table(ext_rows, ["file", "rows", "Favour / Against / None", "median words", "stance target", "domain"]), "",
         "Label mappings: SemEval favor/against/none; Dhoroni Positive/Negative/Neutral (Problematic dropped); "
         "MultiClimate support/oppose/neutral (**oppose = negative sentiment toward climate change itself, so the Against "
         "mapping is low-confidence**); Swami FAVOR/AGAINST/NONE are about **demonetisation**, not climate.", "",
         "## 3. Class distribution", "",
         f"![class distribution](fig_class_distribution.png)", "",
         "* **Against is the rare class everywhere that matters** "
         f"({S['english_merged']['class_pct']['Against']}% of the English merged set; "
         f"{ext_stats.get('semeval2016_climate_en', {}).get('classes', {}).get('Against', 'n/a')} of "
         f"{ext_stats.get('semeval2016_climate_en', {}).get('n', 'n/a')} SemEval tweets). "
         "Macro-F1 weights Against as much as Favour, so ignoring it is expensive.",
         f"* The pilot corpus is balanced **by construction** (50/50/50, 75 Hindi + 75 Bengali); that is not a finding.", "",
         "### Class mix by source (English merged)", "", "![by source](fig_class_by_source.png)", "",
         md_table([[s, len(d), *[int((d.stance == lab).sum()) for lab in LABELS]] for s, d in eng.groupby("source")],
                  ["source", "rows", *LABELS]), "",
         "### Pilot: class x language and script", "", "![pilot lang](fig_pilot_language_class.png)", "",
         "Script-based language ID on the pilot (rows = intended language, columns = detected script):", "",
         pilot_script.to_markdown(), "",
         "Romanised Hindi/Bengali and English are both `latin` by script, so script ID alone cannot separate them.", "",
         "### Pilot: phenomenon x class", "", pilot_phen.to_markdown(), "",
         "## 4. Annotator disagreement (GWStance)", "",
         f"* {extra['ties']} of {extra['gwsd_total_with_ties']} GWSD sentences in the supplied file "
         f"({extra['ties_pct_of_gwsd']}%) have **no annotator majority** ({extra['ties_4_4']} are 4-4 splits, "
         f"{extra['ties_3_3_2']} are 3-3-2). They are dropped from training/evaluation and kept in "
         "`data/gwstance_ambiguous_ties.csv`.",
         f"* Among the {len(gw)} GWSD rows that kept a label, agreement is a median of "
         f"{gw.agreement.median():.3f} (min {gw.agreement.min():.3f}). "
         "Genuine human disagreement bounds how crisp this task can be; it is not a formal ceiling on macro-F1.",
         f"* SemEval rows ({int((eng.label_source == 'published_gold').sum())}) carry one published gold label and no agreement score.", "",
         "## 5. Text length", "", "![length](fig_length_distribution.png)", "",
         "## 6. Vocabulary and hapax rate", "",
         f"* English merged: vocab {S['english_merged']['vocab']}, **hapax rate {S['english_merged']['hapax_rate_pct']}%**.",
         f"* Pilot (Hindi+Bengali+romanised): vocab {S['pilot']['vocab']}, **hapax rate {S['pilot']['hapax_rate_pct']}%** "
         f"over only {S['pilot']['tokens']} tokens.",
         "* A high hapax rate means most word types are seen once: a word-level model has no statistics for them. "
         "Character n-grams share sub-word pieces across spelling variants, which is why `baseline.py` uses char_wb 2-5.", "",
         "## 7. Frequent tokens", "",
         "**English merged** (English stop-words removed): " + ", ".join(f"{w} ({c})" for w, c in frequent(C["english_merged"])), "",
         "**Pilot** (no stop-word list, so function words appear): " +
         ", ".join(f"{w} ({c})" for w, c in frequent(C["pilot"], drop_english_stop=False)), "",
         "## 8. Class-discriminative tokens", "", "![discriminative](fig_discriminative_tokens.png)", "",
         "English merged (smoothed log-odds, min count 8): " +
         "; ".join(f"**{lab}**: " + ", ".join(w for w, _ in disc[lab][:6]) for lab in LABELS), "",
         "Pilot (min count 3, pooled Hindi+Bengali): " +
         "; ".join(f"**{lab}**: " + ", ".join(w for w, _ in disc_pilot[lab][:6]) for lab in LABELS), "",
         artefact,
         f"* **{extra['favour_against_vocab_overlap_pct']}% of the vocabulary of Against sentences also occurs in Favour sentences** "
         "(English merged): the classes are separated by framing, negation and sarcasm, not by a disjoint word list.", "",
         "## 9. Majority-class floor", "",
         md_table([[S[k]["name"], S[k]["majority_floor_macro_f1"]] for k in ("english_merged", "english_highconf", "pilot")],
                  ["dataset", "macro-F1 of always predicting the majority class"]), "",
         "Accuracy would look respectable (the majority share) while macro-F1 is near 0.2: this is why macro-F1 is the metric.", ""]
    open(OUT + "eda_report.md", "w", encoding="utf-8").write("\n".join(L))

    print(f"majority-class macro-F1 floor: english_merged={S['english_merged']['majority_floor_macro_f1']} "
          f"highconf={S['english_highconf']['majority_floor_macro_f1']} pilot={S['pilot']['majority_floor_macro_f1']}")
    print(f"hapax rate: english={S['english_merged']['hapax_rate_pct']}% pilot={S['pilot']['hapax_rate_pct']}%")
    print("wrote outputs/eda_report.md, eda_stats.json, fig_*.png")


def selfcheck():
    assert abs(majority_floor(["Favour", "Favour", "Against"]) - (0.8 + 0 + 0) / 3) < 1e-9  # F1_Favour=0.8, others 0
    df = pd.DataFrame({"comment": ["a b a", "a c", "d"], "stance": ["Favour", "Against", "None"]})
    s, _, counts = stats(df, "toy")
    assert s["vocab"] == 4 and s["hapax"] == 3 and s["hapax_rate_pct"] == 75.0  # a:3 b:1 c:1 d:1
    assert counts["a"] == 3
    print("eda.py self-check passed")


if __name__ == "__main__":
    selfcheck()
    main()
