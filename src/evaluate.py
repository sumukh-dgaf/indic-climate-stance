"""Round 2 evaluation from outputs/preds/ only -> outputs/round2_results.md/.json, fig_r2_*.png, error_examples.csv.

Every number in the report, README and deck is read from outputs/round2_results.json, which this script writes.
Comment text for the error examples is joined from the evaluation data by id (prediction files carry no text).
Run: python src/evaluate.py          Self-check only: python src/evaluate.py --check
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CONFIGS, EVAL_SETS, LABELS, SYNTHETIC, synthetic_verified  # noqa: E402

PREDS, OUT = "outputs/preds", "outputs/"
NAME = re.compile(r"^(?P<model>.+?)_(?P<config>" + "|".join(CONFIGS) + r")_(?P<evalset>" + "|".join(EVAL_SETS) + r")\.csv$")
MODEL_NAMES = {"tfidf": "TF-IDF + LogReg", "muril": "MuRIL", "indicbert": "IndicBERT v2", "xlm-r": "XLM-R"}
MODEL_ORDER = ["tfidf", "muril", "indicbert", "xlm-r"]
SET_NAMES = {"english_test": "English test", "translated_hi_test": "Translated Hindi test",
             "translated_bn_test": "Translated Bengali test", "synthetic_test": "Synthetic hi+bn test", "pilot": "Pilot"}
FAIR_SETS = ["translated_hi_test", "translated_bn_test", "pilot"]  # sets no configuration was built from the same generator as
SEED, N_BOOT = 42, 1000


def macro(gold, pred):
    return float(f1_score(gold, pred, average="macro", labels=LABELS, zero_division=0))


def bootstrap_ci(gold, pred, n=N_BOOT, seed=SEED):
    """95% percentile interval of macro-F1 over n resamples of the test rows (with replacement)."""
    gold, pred = np.asarray(gold), np.asarray(pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gold), size=(n, len(gold)))
    scores = [macro(gold[i], pred[i]) for i in idx]
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def score(df):
    p, r, f, s = precision_recall_fscore_support(df.gold, df.pred, labels=LABELS, zero_division=0)
    majority = df.gold.value_counts().idxmax()
    out = {"n": len(df), "macro_f1": round(macro(df.gold, df.pred), 4), "ci95": bootstrap_ci(df.gold, df.pred),
           "accuracy": round(float(accuracy_score(df.gold, df.pred)), 4),
           "per_class": {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4),
                               "support": int(s[i])} for i, lab in enumerate(LABELS)},
           "majority_class": majority, "floor_macro_f1": round(macro(df.gold, [majority] * len(df)), 4),
           "confusion": confusion_matrix(df.gold, df.pred, labels=LABELS).tolist(),
           "predicted_counts": df.pred.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict()}
    langs = sorted(set(df.language) - {""})
    if len(langs) > 1:
        out["macro_f1_by_language"] = {l: round(macro(g.gold, g.pred), 4) for l, g in df.groupby("language")}
    if (df.phenomenon != "").any():
        out["accuracy_by_phenomenon"] = {ph: {"accuracy": round(float((g.gold == g.pred).mean()), 4), "n": len(g)}
                                         for ph, g in df.groupby("phenomenon")}
    return out


def current_gold(df, stance_by_id):
    """Score synthetic-test predictions against the CURRENT labels: the team review (apply_review.py) may have
    corrected them after the prediction files were written, and those files keep the labels from prediction time."""
    df = df.copy()
    df["gold"] = df.id.map(stance_by_id)
    assert df.gold.notna().all(), "synthetic test prediction ids missing from the synthetic CSV"
    return df


def load_preds(folder=PREDS):
    runs = {}
    syn = pd.read_csv(SYNTHETIC, dtype=str, keep_default_na=False) if os.path.exists(SYNTHETIC) else None
    stance_by_id = dict(zip(syn.id, syn.stance)) if syn is not None else None
    for path in sorted(glob.glob(f"{folder}/*.csv")):
        m = NAME.match(os.path.basename(path))
        if not m:
            print(f"skipping unrecognised file {path}")
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)  # "None" is a label, never NaN
        if m["evalset"] == "synthetic_test" and stance_by_id is not None:
            df = current_gold(df, stance_by_id)
        assert set(df.gold) <= set(LABELS) and set(df.pred) <= set(LABELS), path
        runs.setdefault(f"{m['model']}_{m['config']}", {"model": m["model"], "config": m["config"], "preds": {}})
        runs[f"{m['model']}_{m['config']}"]["preds"][m["evalset"]] = df
    return runs


def pick_best(R):
    cand = {n: np.mean([r["eval"][s]["macro_f1"] for s in FAIR_SETS]) for n, r in R.items() if all(s in r["eval"] for s in FAIR_SETS)}
    return max(cand, key=cand.get) if cand else None


def ordered(names, R):
    return sorted(names, key=lambda n: (MODEL_ORDER.index(R[n]["model"]) if R[n]["model"] in MODEL_ORDER else 9, R[n]["config"]))


def conf_grid(R, best, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = [s for s in EVAL_SETS if s in R[best]["eval"]]
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.6))
    for ax, s in zip(np.atleast_1d(axes), sets):
        cm = np.array(R[best]["eval"][s]["confusion"])
        pct = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if pct[i, j] > 0.5 else "black")
        ax.set_xticks(range(3), LABELS, fontsize=8)
        ax.set_yticks(range(3), LABELS, fontsize=8)
        ax.set_title(SET_NAMES[s], fontsize=9)
        ax.set_xlabel("predicted", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("true")
    fig.suptitle(f"Confusion matrices: {label(R, best)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bars(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ordered(R, R)
    sets = [s for s in EVAL_SETS if any(s in R[n]["eval"] for n in names)]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    w = 0.8 / max(len(names), 1)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(names), 2)))
    for k, n in enumerate(names):
        xs, ys, lo, hi = [], [], [], []
        for i, s in enumerate(sets):
            if s in R[n]["eval"]:
                e = R[n]["eval"][s]
                xs.append(i + k * w - 0.4 + w / 2)
                ys.append(e["macro_f1"])
                lo.append(e["macro_f1"] - e["ci95"][0])
                hi.append(e["ci95"][1] - e["macro_f1"])
        ax.bar(xs, ys, w, yerr=[lo, hi], capsize=2, label=label(R, n), color=colors[k])
    for i, s in enumerate(sets):
        floor = next(R[n]["eval"][s]["floor_macro_f1"] for n in names if s in R[n]["eval"])
        ax.hlines(floor, i - 0.4, i + 0.4, colors="black", linestyles="--", linewidth=1)
    ax.set_xticks(range(len(sets)), [SET_NAMES[s] for s in sets], fontsize=9)
    ax.set_ylabel("macro-F1 (95% bootstrap CI)")
    ax.set_ylim(0, 1)
    ax.set_title("Macro-F1 by model, configuration and test set (dashed: majority-class floor)", loc="left", fontsize=11)
    ax.legend(fontsize=8, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def label(R, n):
    return f"{MODEL_NAMES.get(R[n]['model'], R[n]['model'])} {R[n]['config'].split('_')[0]}"


def error_examples(runs, best, k=15, seed=SEED):
    """Up to k misclassified rows of the best run from the synthetic test + pilot, round-robin over phenomena."""
    from data import eval_sets
    text = {}
    for d in eval_sets().values():
        text.update(dict(zip(d.id, d.comment)))
    rows = []
    for s in ["synthetic_test", "pilot"]:  # our own data only: translated/English rows carry GWStance text (no licence)
        df = runs[best]["preds"].get(s)
        if df is not None:
            rows.append(df[df.gold != df.pred].assign(evalset=s))
    wrong = pd.concat(rows).sample(frac=1, random_state=seed) if rows else pd.DataFrame()
    if wrong.empty:
        return wrong
    groups = [g for _, g in wrong.groupby("phenomenon")]
    pick, i = [], 0
    while len(pick) < k and any(i < len(g) for g in groups):
        pick += [g.iloc[i] for g in groups if i < len(g)][: k - len(pick)]
        i += 1
    out = pd.DataFrame(pick)
    out["comment"] = out.id.map(text)
    out["team_notes"] = ""
    return out[["phenomenon", "evalset", "id", "language", "comment", "gold", "pred", "p_favour", "p_against", "p_none",
                "team_notes"]].sort_values(["phenomenon", "evalset"])


def keep_annotated(new, runs, best, path):
    """If the team has written team_notes, keep those same rows and notes; refresh gold (the review may have changed it)
    and report rows whose gold changed so the notes can be rechecked."""
    if not os.path.exists(path):
        return new
    old = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")  # Excel adds a BOM
    if "team_notes" not in old or not (old.team_notes.str.strip() != "").any():
        return new
    out = old.copy()
    for s, df in runs[best]["preds"].items():  # predictions never change; only the reviewed gold label can
        hit = (out.evalset == s) & out.id.isin(df.id)
        out.loc[hit, "gold"] = out.loc[hit, "id"].map(dict(zip(df.id, df.gold)))
    changed = out[out.gold != old.gold]
    if len(changed):
        print(f"!! kept the team's annotated error_examples.csv; gold label changed by the review for {len(changed)} rows "
              f"(recheck their team_notes): " + ", ".join(f"{r.id} {o}->{r.gold}" for (_, r), o in zip(changed.iterrows(), old.gold[changed.index])))
    if (out.gold == out.pred).any():
        print("!! now predicted correctly after the review: " + ", ".join(out[out.gold == out.pred].id))
    return out


def dataset_stats():
    """Sizes and class distributions of every data family, read from the data files the runs used."""
    from data import ENG, PILOT, SYN_STATS, SYNTHETIC, TRANSLATED, synthetic_complete
    dist = lambda d: {**{lab: int((d.stance == lab).sum()) for lab in LABELS}, "total": len(d)}  # noqa: E731
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)  # noqa: E731
    out = {"english": {s: dist(read(ENG.format(s))) for s in ("train", "val", "test") if os.path.exists(ENG.format(s))},
           "translated": {f"{l}_{s}": dist(read(TRANSLATED.format(l, s))) for l in ("hi", "bn") for s in ("train", "val", "test")
                          if os.path.exists(TRANSLATED.format(l, s))},
           "pilot": dist(read(PILOT))}
    if synthetic_complete():
        syn = read(SYNTHETIC)
        out["synthetic"] = {s: dist(syn[syn.split == s]) for s in ("train", "val", "test")}
        out["synthetic_by_language"] = {l: dist(g) for l, g in syn.groupby("language")}
        out["synthetic_by_phenomenon"] = syn.phenomenon.value_counts().to_dict()
        out["synthetic_generation"] = json.load(open(SYN_STATS, encoding="utf-8"))
    for name, path in (("synthetic_review", "data/review/review_stats.json"),
                       ("translation_review", "data/review/translation_review_stats.json")):
        if os.path.exists(path):
            out[name] = json.load(open(path, encoding="utf-8"))
    return out


def main():
    runs = load_preds()
    assert runs, "no prediction files in outputs/preds/: run src/run_all.py first"
    R = {n: {"model": r["model"], "config": r["config"], "eval": {s: score(df) for s, df in r["preds"].items()}}
         for n, r in runs.items()}
    meta = {}
    for n in R:
        p = f"outputs/runs/{n}.json"
        if os.path.exists(p):
            j = json.load(open(p))
            R[n]["run"] = {k: j.get(k) for k in ("n_train", "n_train_by_language", "best_val_macro_f1", "best_epoch",
                                                  "training_minutes", "gpu", "device")}
    meta["synthetic_test_labels"] = synthetic_verified()
    if os.path.exists("data/translated/translation_info.json"):
        meta["translation"] = json.load(open("data/translated/translation_info.json"))
    best = pick_best(R)
    res = {"runs": R, "best_run": best, "dataset": dataset_stats(),
           "best_rule": "highest mean macro-F1 on translated Hindi, translated Bengali and pilot test sets (the synthetic "
                        "test set is excluded because configuration C trains on data from the same generator)",
           "eval_set_sizes": {s: next(R[n]["eval"][s]["n"] for n in R if s in R[n]["eval"])
                              for s in EVAL_SETS if any(s in R[n]["eval"] for n in R)},
           "bootstrap": {"resamples": N_BOOT, "seed": SEED, "interval": "95% percentile"}, **meta,
           "limitations": [
               "The synthetic test set comes from the same generator as the synthetic training data, so configuration C's "
               "score on it is optimistic; configurations A and B, which never see synthetic data, are the fairer measure of transfer.",
               "No result here is a score on real Hindi/Bengali social-media comments: Indic test sets are machine-translated "
               "English (GWSD news sentences and SemEval tweets), synthetic (LLM-generated) comments, or the 150-row synthetic pilot.",
               "Translated test labels are the English labels carried over; translation can change stance (team review pending).",
               "The English test set has few Against rows, so per-class Against scores on it are noisy.",
               "The judge filter keeps only synthetic comments that a second LLM labels as intended, removing the most "
               "ambiguous ones (most of all sarcasm), so the synthetic test set is easier than real comments would be.",
               "The 150-row pilot is also synthetic comment-style text (team-authored), so it partly favours configuration C too.",
               "Single training seed; differences inside overlapping confidence intervals are not established."]}
    if res.get("synthetic_test_labels") != "yes":
        res["limitations"].append("Synthetic test labels are not yet human-verified (data/review/synthetic_test_review.csv).")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(OUT + "round2_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    bars(R, OUT + "fig_r2_macro_f1.png")
    if best:
        conf_grid(R, best, OUT + "fig_r2_confusion_best.png")
        ex = keep_annotated(error_examples(runs, best), runs, best, OUT + "error_examples.csv")
        ex.to_csv(OUT + "error_examples.csv", index=False)
    write_md(res)
    update_readme(res)
    print(f"wrote outputs/round2_results.md/.json, fig_r2_*.png, error_examples.csv; best run: {best}")


def fmt(e):
    return f"{e['macro_f1']:.3f} [{e['ci95'][0]:.2f}-{e['ci95'][1]:.2f}]"


def write_md(res):
    R, best = res["runs"], res["best_run"]
    names = ordered(R, R)
    sets = list(res["eval_set_sizes"])
    verified = res.get("synthetic_test_labels") == "yes"
    head = lambda s: SET_NAMES[s] + ("" if s != "synthetic_test" or verified else " *")  # noqa: E731
    L = ["# Round 2 results (generated by src/evaluate.py from outputs/preds/)", "",
         "Macro-F1 with 95% bootstrap confidence interval (1,000 resamples). Floor = always predicting the test set's "
         "majority class.", "", "## Limitations (read first)", ""] + [f"* {x}" for x in res["limitations"]] + [
         "", "## Main table: macro-F1 [95% CI]", "",
         "| Model | Config | " + " | ".join(f"{head(s)} (n={res['eval_set_sizes'][s]})" for s in sets) + " |",
         "|---|---|" + "---|" * len(sets)]
    for n in names:
        L.append(f"| {MODEL_NAMES.get(R[n]['model'], R[n]['model'])} | {R[n]['config']} | " +
                 " | ".join(fmt(R[n]["eval"][s]) if s in R[n]["eval"] else "not run" for s in sets) + " |")
    floors = {s: next(R[n]["eval"][s]["floor_macro_f1"] for n in names if s in R[n]["eval"]) for s in sets}
    L.append("| Majority floor | - | " + " | ".join(f"{floors[s]:.3f}" for s in sets) + " |")
    if not verified:
        L += ["", "\\* synthetic test labels not yet human-verified."]
    L += ["", "![macro-F1](fig_r2_macro_f1.png)", "", "## Accuracy", "",
          "| Model | Config | " + " | ".join(SET_NAMES[s] for s in sets) + " |", "|---|---|" + "---|" * len(sets)]
    for n in names:
        L.append(f"| {MODEL_NAMES.get(R[n]['model'])} | {R[n]['config']} | " +
                 " | ".join(f"{R[n]['eval'][s]['accuracy']:.3f}" if s in R[n]["eval"] else "-" for s in sets) + " |")

    # the three comparisons
    L += ["", "## Comparison 1: models on the same data (macro-F1)", ""]
    for c in CONFIGS:
        ns = [n for n in names if R[n]["config"] == c]
        if ns:
            L.append(f"* **{c}**: " + "; ".join(
                f"{SET_NAMES[s]}: " + ", ".join(f"{MODEL_NAMES.get(R[n]['model'])} {R[n]['eval'][s]['macro_f1']:.3f}"
                                                for n in ns if s in R[n]["eval"]) for s in sets))
    L += ["", "## Comparison 2: what translated (B) and synthetic (C) data add (macro-F1 change vs A)", ""]
    for m in MODEL_ORDER:
        a = f"{m}_A_english"
        if a not in R:
            continue
        for c in ("B_translated", "C_full"):
            n = f"{m}_{c}"
            if n in R:
                L.append(f"* {MODEL_NAMES[m]} {c} vs A: " + ", ".join(
                    f"{SET_NAMES[s]} {R[n]['eval'][s]['macro_f1'] - R[a]['eval'][s]['macro_f1']:+.3f}"
                    for s in sets if s in R[n]["eval"] and s in R[a]["eval"]))
    L += ["", "## Comparison 3: English test vs Indic test sets (cross-lingual loss)", "",
          "| Model | Config | English | mean of translated hi/bn + pilot | difference |", "|---|---|---|---|---|"]
    for n in names:
        e = R[n]["eval"]
        if "english_test" in e and all(s in e for s in FAIR_SETS):
            ind = np.mean([e[s]["macro_f1"] for s in FAIR_SETS])
            L.append(f"| {MODEL_NAMES.get(R[n]['model'])} | {R[n]['config']} | {e['english_test']['macro_f1']:.3f} | "
                     f"{ind:.3f} | {ind - e['english_test']['macro_f1']:+.3f} |")
    if best:
        b = R[best]
        L += ["", f"## Best run: {label(R, best)} ({best})", "", f"Selection rule: {res['best_rule']}.", "",
              "| Test set | Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|---|"]
        for s in sets:
            if s in b["eval"]:
                for lab in LABELS:
                    pc = b["eval"][s]["per_class"][lab]
                    L.append(f"| {SET_NAMES[s]} | {lab} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | {pc['support']} |")
        L += ["", "![confusion](fig_r2_confusion_best.png)", "", "### Macro-F1 by language", ""]
        for s in sets:
            if "macro_f1_by_language" in b["eval"].get(s, {}):
                L.append(f"* {SET_NAMES[s]}: " + ", ".join(f"{l} {v:.3f}" for l, v in b["eval"][s]["macro_f1_by_language"].items()))
        for s in ("translated_hi_test", "translated_bn_test"):
            if s in b["eval"]:
                L.append(f"* {SET_NAMES[s]}: {b['eval'][s]['macro_f1']:.3f}")
        L += ["", "### Accuracy by phenomenon", "", "| Test set | Phenomenon | Accuracy | n |", "|---|---|---|---|"]
        for s in sets:
            for ph, v in b["eval"].get(s, {}).get("accuracy_by_phenomenon", {}).items():
                L.append(f"| {SET_NAMES[s]} | {ph} | {v['accuracy']:.3f} | {v['n']} |")
        L += ["", "Misclassified examples for error analysis: `outputs/error_examples.csv` (15 rows, grouped by phenomenon)."]
    L += ["", "## Training runs", "", "| Run | Train rows | Best val macro-F1 | Best epoch | Minutes | Device |", "|---|---|---|---|---|---|"]
    for n in names:
        r = R[n].get("run")
        if r:
            L.append(f"| {n} | {r['n_train']} | {r['best_val_macro_f1']:.3f} | {r.get('best_epoch') or '-'} | "
                     f"{r['training_minutes']:.1f} | {r.get('gpu') or r.get('device') or 'cpu'} |")
    L.append("\nValidation scores are not comparable across configurations: each configuration selects on its own validation data.")
    open(OUT + "round2_results.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


def update_readme(res, path="README.md"):
    """Rewrite the generated results table between the RESULTS markers of the README."""
    R = res["runs"]
    names, sets = ordered(R, R), list(res["eval_set_sizes"])
    verified = res.get("synthetic_test_labels") == "yes"
    L = ["Test macro-F1 (95% bootstrap CIs in `outputs/round2_results.md`):", "",
         "| Model | Config | " + " | ".join(SET_NAMES[s] + ("" if s != "synthetic_test" or verified else " *") for s in sets) + " |",
         "|---|---|" + "---|" * len(sets)]
    for n in names:
        L.append(f"| {MODEL_NAMES.get(R[n]['model'], R[n]['model'])} | {R[n]['config']} | " +
                 " | ".join(f"{R[n]['eval'][s]['macro_f1']:.3f}" if s in R[n]["eval"] else "not run" for s in sets) + " |")
    L.append("| Majority floor | - | " + " | ".join(
        f"{next(R[n]['eval'][s]['floor_macro_f1'] for n in names if s in R[n]['eval']):.3f}" for s in sets) + " |")
    if res["best_run"]:
        L += ["", f"Best run (rule: {res['best_rule']}): **{res['best_run']}**."]
    if not verified:
        L += ["", "\\* synthetic test labels not yet human-verified."]
    text = open(path, encoding="utf-8").read()
    a, b = text.index("<!-- RESULTS:START -->"), text.index("<!-- RESULTS:END -->")
    open(path, "w", encoding="utf-8").write(text[:a] + "<!-- RESULTS:START -->\n" + "\n".join(L) + "\n" + text[b:])


def selfcheck():
    gold = np.array(LABELS * 20)
    assert macro(gold, gold) == 1.0
    lo, hi = bootstrap_ci(gold, gold)
    assert lo == hi == 1.0
    noisy = gold.copy()
    noisy[:15] = "None"
    lo, hi = bootstrap_ci(gold, noisy)
    assert lo < macro(gold, noisy) < hi
    df = pd.DataFrame({"gold": gold, "pred": noisy, "language": ["hi", "bn"] * 30, "phenomenon": ["plain"] * 60})
    s = score(df)
    assert s["floor_macro_f1"] == round(macro(gold, [s["majority_class"]] * 60), 4) and "macro_f1_by_language" in s
    assert NAME.match("indicbert_C_full_translated_hi_test.csv")["evalset"] == "translated_hi_test"
    import tempfile
    with tempfile.TemporaryDirectory() as t:  # annotated examples keep their notes; gold follows the reviewed labels
        f = os.path.join(t, "ex.csv")
        pd.DataFrame({"evalset": ["pilot"], "id": ["a"], "gold": ["Favour"], "pred": ["None"], "team_notes": ["sarcasm"]}).to_csv(f, index=False)
        k = keep_annotated(None, {"r": {"preds": {"pilot": pd.DataFrame({"id": ["a"], "gold": ["Against"]})}}}, "r", f)
        assert k.team_notes.tolist() == ["sarcasm"] and k.gold.tolist() == ["Against"]
    assert NAME.match("xlm-r_A_english_pilot.csv")["model"] == "xlm-r"
    # reviewed labels must win over the labels stored in a prediction file
    p = pd.DataFrame({"id": ["a", "b"], "gold": ["Favour", "None"], "pred": ["Favour", "Against"]})
    assert current_gold(p, {"a": "Favour", "b": "Against"}).gold.tolist() == ["Favour", "Against"]
    print("evaluate.py self-check passed")


if __name__ == "__main__":
    selfcheck()
    if "--check" not in sys.argv:
        main()
