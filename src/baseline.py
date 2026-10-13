"""TF-IDF baselines vs a majority-class floor -> outputs/baseline_results.md/.json + confusion-matrix PNGs.

Experiments
  A  English merged (2,408): stratified 5-fold CV, feature-set x model grid, class-weight ablation
  B  English high-confidence (1,686): same config, secondary
  C  Pilot hi+bn (150, synthetic, DEV ONLY): stratified 5-fold CV
  D  Cross-lingual transfer: train on ALL English, test on the pilot (the situation of the shared task)
Run: python src/baseline.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import LABELS, load, load_english, validate  # noqa: E402
from preprocess import clean, script_id, tokenize  # noqa: E402

SEED = 42
OUT = "outputs/"
PILOT_TAG = "pilot corpus, development only - not comparable to official-set results"


def features(kind):
    # word 1-2 grams use our Indic-aware tokenizer; char_wb 2-5 grams stay inside word boundaries
    word = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    return {"word": word, "char": char, "word+char": FeatureUnion([("word", word), ("char", char)])}[kind]


def model(feat, clf, balanced=True):
    cw = "balanced" if balanced else None  # 'balanced' weights each class by n / (3 * n_class)
    est = {"LogReg": LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED),
           "LinearSVC": LinearSVC(class_weight=cw, random_state=SEED),
           "Dummy(majority)": DummyClassifier(strategy="most_frequent")}[clf]
    return Pipeline([("features", features(feat)), ("clf", est)])


def cv(make, X, y, k=5):
    """Stratified k-fold. Returns per-fold macro-F1/accuracy and the pooled out-of-fold predictions."""
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    f1s, accs, pred = [], [], np.empty(len(y), dtype=object)
    for tr, te in folds.split(X, y):
        m = make().fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        pred[te] = p
        f1s.append(f1_score(y.iloc[te], p, average="macro", labels=LABELS, zero_division=0))
        accs.append(accuracy_score(y.iloc[te], p))
    return {"macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)), "accuracy": float(np.mean(accs)),
            "pooled_macro_f1": float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0))}, pred


def conf_fig(y, pred, title, path):
    cm = confusion_matrix(y, pred, labels=LABELS)
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{100 * pct[i, j]:.0f}%", ha="center", va="center", fontsize=12,
                    color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return cm


def rows_table(res):
    return "\n".join(["| features | model | macro-F1 (mean +- sd over folds) | accuracy |", "|---|---|---|---|"] +
                     [f"| {r['features']} | {r['model']} | {r['macro_f1']:.3f} +- {r['macro_f1_std']:.3f} | {r['accuracy']:.3f} |"
                      for r in res])


def main():
    os.makedirs(OUT, exist_ok=True)
    eng = load_english()
    high = validate(load("data/english_stance_highconf.csv"), "english_stance_highconf.csv")
    pilot = validate(load("data/pilot_indic_stance.csv", default_source="team-authored pilot (synthetic, dev only)"),
                     "pilot_indic_stance.csv")
    for d in (eng, high, pilot):
        d["x"] = d.comment.map(clean)
    R = {}

    # ---------------- A. English merged
    grid = []
    preds = {}
    for feat in ("word", "char", "word+char"):
        for clf in ("LogReg", "LinearSVC"):
            r, p = cv(lambda: model(feat, clf), eng.x, eng.stance)
            grid.append({"features": feat, "model": clf, **r})
            preds[(feat, clf)] = p
            print(f"A  {feat:10s} {clf:10s} macro-F1 {r['macro_f1']:.3f} +- {r['macro_f1_std']:.3f}  acc {r['accuracy']:.3f}")
    rd, _ = cv(lambda: model("word+char", "Dummy(majority)"), eng.x, eng.stance)
    grid.append({"features": "-", "model": "Dummy(majority)", **rd})
    print(f"A  floor       Dummy      macro-F1 {rd['macro_f1']:.3f}  acc {rd['accuracy']:.3f}")
    R["A_english_cv"] = grid

    best = max(grid[:-1], key=lambda r: r["macro_f1"])
    bp = preds[(best["features"], best["model"])]
    cm = conf_fig(eng.stance, bp, f"English CV: {best['features']} + {best['model']}", OUT + "fig_confusion_english_cv.png")
    report = classification_report(eng.stance, bp, labels=LABELS, digits=3, zero_division=0)
    by_source = {s: float(f1_score(g.stance, bp[g.index], average="macro", labels=LABELS, zero_division=0))
                 for s, g in eng.groupby("source")}
    # error analysis: largest off-diagonal cell, with examples
    off = [(cm[i, j], LABELS[i], LABELS[j]) for i in range(3) for j in range(3) if i != j]
    n_err, t_lab, p_lab = max(off)
    wrong = eng[(eng.stance == t_lab) & (pd.Series(bp, index=eng.index) == p_lab)].comment.head(3).tolist()
    against_i = LABELS.index("Against")
    against_to_none = int(cm[against_i, LABELS.index("None")])
    R["A_best"] = {"config": f"{best['features']} + {best['model']}", "report": report, "confusion": cm.tolist(),
                   "macro_f1_by_source": by_source, "largest_error": {"true": t_lab, "pred": p_lab, "n": int(n_err), "examples": wrong},
                   "against_recall": float(cm[against_i, against_i] / cm[against_i].sum()), "against_to_none": against_to_none}

    # class-weight ablation (same features/model as best, only class_weight changes)
    abl = {}
    for bal in (True, False):
        r, _ = cv(lambda: model(best["features"], best["model"], balanced=bal), eng.x, eng.stance)
        abl["balanced" if bal else "none"] = r
        print(f"A  class_weight={'balanced' if bal else 'None':8s} macro-F1 {r['macro_f1']:.3f}")
    # recall of the rare class with / without weighting (pooled out-of-fold)
    for bal in (True, False):
        _, p = cv(lambda: model(best["features"], best["model"], balanced=bal), eng.x, eng.stance)
        rep = classification_report(eng.stance, p, labels=LABELS, output_dict=True, zero_division=0)
        abl["balanced" if bal else "none"]["against_recall"] = rep["Against"]["recall"]
    R["A_class_weight_ablation"] = abl

    # ---------------- B. high-confidence
    rb, _ = cv(lambda: model(best["features"], best["model"]), high.x, high.stance)
    rbd, _ = cv(lambda: model("word+char", "Dummy(majority)"), high.x, high.stance)
    R["B_highconf_cv"] = {"config": R["A_best"]["config"], **rb, "dummy_macro_f1": rbd["macro_f1"]}
    print(f"B  highconf {R['A_best']['config']} macro-F1 {rb['macro_f1']:.3f}  (floor {rbd['macro_f1']:.3f})")

    # ---------------- C. pilot CV (synthetic, dev only)
    pil = []
    for feat in ("word", "char", "word+char"):
        for clf in ("LogReg", "LinearSVC"):
            r, _ = cv(lambda: model(feat, clf), pilot.x, pilot.stance)
            pil.append({"features": feat, "model": clf, **r})
            print(f"C  {feat:10s} {clf:10s} macro-F1 {r['macro_f1']:.3f}  [{PILOT_TAG}]")
    rpd, _ = cv(lambda: model("word+char", "Dummy(majority)"), pilot.x, pilot.stance)
    pil.append({"features": "-", "model": "Dummy(majority)", **rpd})
    R["C_pilot_cv"] = pil

    # ---------------- D. cross-lingual transfer: English -> pilot
    m = model(best["features"], best["model"]).fit(eng.x, eng.stance)
    pp = m.predict(pilot.x)
    pilot["pred"] = pp
    pilot["script"] = pilot.comment.map(script_id)
    D = {"config": R["A_best"]["config"], "macro_f1": float(f1_score(pilot.stance, pp, average="macro", labels=LABELS, zero_division=0)),
         "accuracy": float(accuracy_score(pilot.stance, pp)),
         "macro_f1_by_language": {l: float(f1_score(g.stance, g.pred, average="macro", labels=LABELS, zero_division=0))
                                  for l, g in pilot.groupby("language")},
         "accuracy_by_script": {s: float((g.stance == g.pred).mean()) for s, g in pilot.groupby("script")},
         "n_by_script": pilot.script.value_counts().to_dict(),
         "predicted_class_counts": pilot.pred.value_counts().reindex(LABELS, fill_value=0).to_dict(),
         "confusion": confusion_matrix(pilot.stance, pp, labels=LABELS).tolist()}
    conf_fig(pilot.stance, pp, "English -> pilot (zero-shot, TF-IDF)", OUT + "fig_confusion_transfer_pilot.png")
    R["D_transfer_to_pilot"] = D
    print(f"D  English->pilot macro-F1 {D['macro_f1']:.3f}  acc {D['accuracy']:.3f}  [{PILOT_TAG}]")
    print("   predicted class counts:", D["predicted_class_counts"], "| accuracy by script:", D["accuracy_by_script"])

    json.dump(R, open(OUT + "baseline_results.json", "w"), indent=1)

    # ---------------- comparison figure
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = [f"{r['features']}\n{r['model']}" for r in grid[:-1]]
    vals = [r["macro_f1"] for r in grid[:-1]]
    errs = [r["macro_f1_std"] for r in grid[:-1]]
    ax.bar(range(len(vals)), vals, yerr=errs, color="#2a9d8f", capsize=4)
    ax.axhline(grid[-1]["macro_f1"], color="#e76f51", ls="--", label=f"majority-class floor ({grid[-1]['macro_f1']:.2f})")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_xticks(range(len(vals)), labels, fontsize=10)
    ax.set_ylabel("macro-F1 (5-fold CV)")
    ax.set_ylim(0, 1)
    ax.set_title("English merged (2,408): TF-IDF baselines", loc="left", fontsize=13)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT + "fig_baseline_comparison.png", dpi=160)
    plt.close(fig)

    # ---------------- markdown
    L = ["# Baseline results (generated by src/baseline.py)", "",
         "TF-IDF (word 1-2 + char_wb 2-5) with LogisticRegression / LinearSVC, `class_weight='balanced'`, "
         f"stratified 5-fold CV, seed {SEED}. Floor = `DummyClassifier(most_frequent)`.", "",
         "> **Caveats.** English CV mixes news sentences (GWSD) with tweets (SemEval); the files have no document ids, so "
         "sentences from one article may sit in different folds, and one conference hashtag (#CFCC15) forms a memorisable "
         f"cluster. Treat A and B as optimistic. Section C/D use the **{PILOT_TAG}** (150 synthetic rows).", "",
         "## A. English merged, 5-fold CV (2,408 rows)", "", rows_table(grid), "",
         "![baselines](fig_baseline_comparison.png)", "",
         f"Best: **{R['A_best']['config']}**. Pooled out-of-fold classification report:", "", "```", report, "```", "",
         "![confusion](fig_confusion_english_cv.png)", "",
         "Macro-F1 of the pooled out-of-fold predictions by source: " +
         ", ".join(f"{s.split(' (')[0]} {v:.3f}" for s, v in by_source.items()) + ".", "",
         "### Error analysis", "",
         f"* Largest confusion: true **{t_lab}** predicted **{p_lab}** ({int(n_err)} sentences). Examples: " +
         " | ".join(f"\"{w[:110]}\"" for w in wrong),
         f"* Against recall {R['A_best']['against_recall']:.2f}; {against_to_none} Against sentences were called None.", "",
         "### Why `class_weight='balanced'`", "",
         f"Same model, only the class weights changed: macro-F1 {abl['balanced']['macro_f1']:.3f} (balanced) vs "
         f"{abl['none']['macro_f1']:.3f} (unweighted); Against recall {abl['balanced']['against_recall']:.2f} vs "
         f"{abl['none']['against_recall']:.2f}. Macro-F1 averages per-class F1 with equal weight, so the rare Against class "
         "counts as much as the two large ones; unweighted training optimises overall error and under-serves it.", "",
         "## B. English high-confidence (1,686 rows, agreement >= 0.75), same config", "",
         f"macro-F1 {rb['macro_f1']:.3f} +- {rb['macro_f1_std']:.3f} (floor {rbd['macro_f1']:.3f}).", "",
         f"## C. Pilot hi+bn, 5-fold CV ({PILOT_TAG})", "", rows_table(pil), "",
         "150 rows with 30 per test fold: the standard deviations are large; use only to check the pipeline runs on Indic text.", "",
         f"## D. Zero-shot transfer: train on English (2,408) -> test on pilot ({PILOT_TAG})", "",
         f"* Config {D['config']}: **macro-F1 {D['macro_f1']:.3f}**, accuracy {D['accuracy']:.3f} "
         f"(majority floor on pilot = 0.167).",
         f"* Macro-F1 by language: " + ", ".join(f"{l} {v:.3f}" for l, v in D["macro_f1_by_language"].items()),
         f"* Accuracy by detected script (n): " +
         ", ".join(f"{s} {D['accuracy_by_script'][s]:.2f} (n={D['n_by_script'][s]})" for s in D["accuracy_by_script"]),
         f"* Predicted class counts: {D['predicted_class_counts']}", "",
         "![transfer](fig_confusion_transfer_pilot.png)", "",
         "A TF-IDF model trained on English has almost no features in common with Devanagari/Bengali text, so it falls "
         "back on a prior. This is the cross-lingual gap that a multilingual encoder (MuRIL, XLM-R) is meant to close.", ""]
    open(OUT + "baseline_results.md", "w", encoding="utf-8").write("\n".join(L))
    print("wrote outputs/baseline_results.md/.json, fig_confusion_*.png, fig_baseline_comparison.png")


def selfcheck():
    # toy 3-class problem: the model must separate it, and the literal label "None" must survive end to end
    X = pd.Series(["good great fine"] * 10 + ["bad awful worse"] * 10 + ["video link sub"] * 10)
    y = pd.Series(["Favour"] * 10 + ["Against"] * 10 + ["None"] * 10)
    r, p = cv(lambda: model("word+char", "LogReg"), X, y)
    assert r["macro_f1"] > 0.95 and "None" in set(p)
    rd, _ = cv(lambda: model("word", "Dummy(majority)"), X, y)
    assert abs(rd["macro_f1"] - 1 / 6) < 0.05  # one class always: that class F1 = 0.5, others 0 -> 0.5 / 3
    # the tokenizer keeps Hindi words whole, so word features are not fragmented
    v = features("word").fit(["जलवायु परिवर्तन"])
    assert "जलवायु" in v.get_feature_names_out() if hasattr(v, "get_feature_names_out") else True
    print("baseline.py self-check passed")


if __name__ == "__main__":
    selfcheck()
    main()
