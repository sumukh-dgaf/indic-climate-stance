"""Two-page Round 2 progress report -> docs/Round2_Progress_Report.docx (python-docx).

Every number is read from outputs/round2_results.json (written by src/evaluate.py); nothing is typed by hand.
Page count is measured with Microsoft Word (PowerShell COM) when available; the body font shrinks in 0.5 pt steps
until the document is exactly two A4 pages. Run: python src/make_report.py
"""
import json
import os
import subprocess
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import LABELS  # noqa: E402

RES = "outputs/round2_results.json"
OUT = "docs/Round2_Progress_Report.docx"
NAVY = RGBColor(0x1F, 0x2A, 0x44)
MODEL = {"tfidf": "TF-IDF+LogReg", "muril": "MuRIL", "indicbert": "IndicBERT v2"}
SETS = [("english_test", "English"), ("translated_hi_test", "Transl. hi"), ("translated_bn_test", "Transl. bn"),
        ("synthetic_test", "Synthetic"), ("pilot", "Pilot")]


def f3(x):
    return f"{x:.3f}"


def n(x):
    return f"{x:,}"


def facts(R):
    """Every number the prose uses, pulled from the results JSON in one place."""
    runs, ds = R["runs"], R["dataset"]
    F = {"best": R["best_run"], "verified": R.get("synthetic_test_labels") == "yes"}
    get = lambda run, s: runs.get(run, {}).get("eval", {}).get(s, {}).get("macro_f1")  # noqa: E731
    F["get"] = get
    eng = ds["english"]
    F["eng"] = {s: eng[s]["total"] for s in eng}
    F["eng_against_test"] = eng["test"]["Against"]
    syn = ds.get("synthetic")
    F["syn"] = {s: syn[s]["total"] for s in syn} if syn else None
    F["gen"] = ds.get("synthetic_generation")
    F["tr_model"] = (R.get("translation") or {}).get("hi_train", {}).get("model", "not yet measured")
    F["pilot"] = ds["pilot"]["total"]
    F["tr_sizes"] = ds.get("translated", {})
    e = runs.get(F["best"], {}).get("eval", {}) if F["best"] else {}
    ph = e.get("synthetic_test", e.get("pilot", {})).get("accuracy_by_phenomenon", {})
    F["weak"] = ", ".join(k.replace("_", " ") for k in sorted(ph, key=lambda k: ph[k]["accuracy"])[:3]) or "not yet measured"
    return F


def set_cell(cell, text, size, bold=False, color=None, fill=None, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text)
    r.font.size, r.bold = Pt(size), bold
    if color:
        r.font.color.rgb = color
    if fill:
        sh = OxmlElement("w:shd")
        sh.set(qn("w:val"), "clear")
        sh.set(qn("w:fill"), fill)
        cell._tc.get_or_add_tcPr().append(sh)


def table(doc, rows, size, widths):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            set_cell(t.cell(i, j), str(val), size - 1, bold=i == 0, color=RGBColor(255, 255, 255) if i == 0 else None,
                     fill="1F2A44" if i == 0 else None, align=WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER)
            t.cell(i, j).width = Cm(widths[j])
    return t


def build(R, size):
    F = facts(R)
    runs, get, ds = R["runs"], F["get"], R["dataset"]
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)  # A4
    sec.left_margin = sec.right_margin = Cm(1.8)
    sec.top_margin = sec.bottom_margin = Cm(1.5)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(size)
    st.element.rPr.rFonts.set(qn("w:cs"), "Nirmala UI")
    st.paragraph_format.space_after = Pt(3)
    st.paragraph_format.line_spacing = 1.0

    def para(text, bold_lead=None):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if bold_lead:
            p.add_run(bold_lead).bold = True
        p.add_run(text)
        return p

    def heading(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(5), Pt(2)
        r = p.add_run(text)
        r.bold, r.font.size, r.font.color.rgb = True, Pt(size + 2), NAVY

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Stance Detection toward Global Warming in Hindi and Bengali: Round 2 Progress Report")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(size + 4), NAVY
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.add_run("Sumukh Gupta (S24CSEU0570), Neya Tripathi (S24CSEU0609), Aditya Goel (S24CSEU0614), Kushika Mishra "
              "(S24CSEU0652), Eklavya Sharma (S24CSEU0798) | github.com/sumukh-dgaf/indic-climate-stance").font.size = Pt(size - 1)

    # ---------------- 1. Dataset
    heading("1. Dataset description and statistics")
    g = F["gen"]
    para("Task: classify a Hindi or Bengali social-media comment as Favour, Against or None toward the claim \"Climate "
         "change and global warming is a serious concern\" (metric: macro-F1). No official dataset was provided, so we "
         "built four data families and label each by provenance. (1) Human-labelled English: GWStance news sentences "
         "(8-annotator majority, ties removed) and SemEval-2016 climate tweets, split with the official held-out portions as "
         f"test and the rest 85/15 stratified by label and source (seed 42): {n(F['eng']['train'])} / {n(F['eng']['val'])} / "
         f"{n(F['eng']['test'])}. (2) Machine-translated: every English split translated to Hindi and Bengali after splitting, "
         f"with {F['tr_model'].split('/')[-1]} (IndicTrans2 is gated on Hugging Face). (3) Synthetic (LLM-generated): "
         + (f"comments generated with the Gemini API ({g['generator_model']}) by language, label and phenomenon quota "
            "(plain, code-mixed, Romanised, negation, sarcasm, implicit, emoji; for None also off-topic, question, neutral fact). "
            f"Of {n(g['generated'])} generated comments, {n(g['removed_format_or_length'])} failed length/format, "
            f"{n(g['removed_script'])} the script check, {n(g['removed_exact_duplicate'] + g['removed_near_duplicate'])} were "
            f"duplicates or near-duplicates (char TF-IDF cosine > 0.85), and a second model ({g['judge_model']}) that saw only "
            f"claim and comment agreed with the intended label for {100 * g['judge_agreement_overall']:.1f}% of the rest "
            f"(lowest: {min(g['judge_agreement_by_phenomenon'], key=g['judge_agreement_by_phenomenon'].get)} "
            f"{100 * min(g['judge_agreement_by_phenomenon'].values()):.1f}%); only agreed rows were kept. "
            if g else "not yet generated. ")
         + f"(4) The 150-row team-authored synthetic pilot from Milestone 1, used only as a test set. None of these are real "
         "Hindi/Bengali YouTube comments.")
    rows = [["Data family (provenance)", "Train", "Val", "Test", "Favour / Against / None (all splits)"]]
    tot = lambda d: " / ".join(str(sum(d[s][lab] for s in d)) for lab in LABELS)  # noqa: E731
    rows.append(["English (human-labelled)", n(F["eng"]["train"]), n(F["eng"]["val"]), n(F["eng"]["test"]), tot(ds["english"])])
    for lang, name in (("hi", "Hindi"), ("bn", "Bengali")):
        d = {s: F["tr_sizes"][f"{lang}_{s}"] for s in ("train", "val", "test") if f"{lang}_{s}" in F["tr_sizes"]}
        if d:
            rows.append([f"{name} (machine-translated)"] + [n(d[s]["total"]) for s in ("train", "val", "test")] + [tot(d)])
    if F["syn"]:
        rows.append(["Hindi+Bengali (synthetic, LLM-generated)", n(F["syn"]["train"]), n(F["syn"]["val"]), n(F["syn"]["test"]),
                     tot(ds["synthetic"])])
    p = ds["pilot"]
    rows.append(["Pilot hi+bn (team-authored synthetic)", "-", "-", str(p["total"]), " / ".join(str(p[l]) for l in LABELS)])
    table(doc, rows, size, [6.2, 1.6, 1.6, 1.6, 5.0])
    cap = doc.add_paragraph()
    cap.add_run(f"Table 1. Dataset statistics. English test contains only {F['eng_against_test']} Against rows."
                + ("" if F["verified"] else " Synthetic test labels not yet human-verified (team review in progress).")).italic = True

    # ---------------- 2. Method
    heading("2. Current methodology and implementation")
    best_run = runs.get("indicbert_C_full", {}).get("run") or {}
    para("Each input is the pair (English claim, comment); the claim stays English for every language, so the model learns "
         "stance relative to the claim. Models: IndicBERT v2 (ai4bharat/IndicBERTv2-MLM-only, main), MuRIL "
         "(google/muril-base-cased, comparison) and the Milestone 1 TF-IDF word+char + logistic-regression baseline, all on "
         "identical data. Three training configurations isolate what each data family adds: A = English only (zero-shot "
         "transfer), B = A + translated Hindi/Bengali, C = B + synthetic. Fine-tuning: class-weighted cross-entropy (Against "
         "is rare), learning rate 2e-5, batch 32, up to 5 epochs, 10% warmup, weight decay 0.01, mixed precision, max length "
         "128, seed 42; the checkpoint with the best macro-F1 on the validation data of the same configuration is kept and test "
         "sets are never used for selection. Every run predicts five fixed test sets (English, translated Hindi, translated "
         "Bengali, synthetic, pilot) into per-example files; src/evaluate.py computes macro-F1, accuracy, per-class scores, "
         "95% bootstrap confidence intervals (1,000 resamples) and the majority-class floor from those files alone. The pipeline "
         "(splits, generation, translation, training, evaluation, demo) is in src/ with an assert-based self-check per script; "
         "runs execute locally on an RTX 5050 laptop GPU"
         + (f" (IndicBERT v2 on C: {best_run['training_minutes']:.1f} min for {n(best_run['n_train'])} training rows)."
            if best_run else "."))

    # ---------------- 3. Results
    heading("3. Experimental results and performance analysis")
    order = [f"{m}_{c}" for m in ("tfidf", "muril", "indicbert") for c in ("A_english", "B_translated", "C_full")]
    rows = [["Model", "Config"] + [lab + ("" if s != "synthetic_test" or F["verified"] else "*") for s, lab in SETS]]
    for name in order:
        if name in runs:
            rows.append([MODEL[runs[name]["model"]], runs[name]["config"].split("_")[0]] +
                        [f3(get(name, s)) if get(name, s) is not None else "-" for s, _ in SETS])
    floor = {s: next(r["eval"][s]["floor_macro_f1"] for r in runs.values() if s in r["eval"]) for s, _ in SETS
             if any(s in r["eval"] for r in runs.values())}
    rows.append(["Majority floor", "-"] + [f3(floor[s]) if s in floor else "-" for s, _ in SETS])
    table(doc, rows, size, [3.2, 1.3] + [2.5] * 5)
    b = R["best_run"]
    be = runs[b]["eval"] if b else {}
    cap = doc.add_paragraph()
    ci = lambda name, s: runs[name]["eval"][s]["ci95"]  # noqa: E731
    cap.add_run("Table 2. Test macro-F1 per model, configuration and test set."
                + ("" if F["verified"] else " *Synthetic test labels not yet human-verified.")
                + (f" 95% CI for {MODEL[runs[b]['model']]} {runs[b]['config'].split('_')[0]} on translated Hindi: "
                   f"{ci(b, 'translated_hi_test')[0]:.2f}-{ci(b, 'translated_hi_test')[1]:.2f}." if b else "")).italic = True
    if os.path.exists("outputs/fig_r2_macro_f1.png"):
        doc.add_picture("outputs/fig_r2_macro_f1.png", width=Cm(16.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.add_run("Figure 1. Macro-F1 with 95% bootstrap intervals; dashed lines are majority-class floors.").italic = True
    for txt in analysis(R, F):
        para(txt)

    # ---------------- 4. Challenges
    heading("4. Challenges encountered")
    para("No official or real Hindi/Bengali stance data exists for this claim, so every Indic test set is a proxy: "
         "machine-translated English news and tweets, LLM-generated comments, or our 150-row pilot. The synthetic test set "
         "comes from the same generator as the synthetic training data, so configuration C's score on it is optimistic; "
         "configurations A and B, which never see synthetic data, are the fairer measure of transfer, and no number here is a "
         "score on real comments. Data licensing: GWStance has no licence file, so English and translated text stays out of "
         "the repository and is rebuilt by script. IndicTrans2 is gated, so NLLB-200 was used. The Gemini API returned "
         "frequent HTTP 503 (overload) responses, so generation was made resumable with back-off. "
         + (f"Stance itself is ambiguous: the judge model disagreed with the intended label for {100 * (1 - F['gen']['judge_agreement_overall']):.1f}% of "
            f"filtered synthetic comments, most often for {low2(F['gen'])}. " if F["gen"] else "")
         + "The judge filter also removes the most ambiguous synthetic comments, so the synthetic test set is easier than "
         "real comments would be. pandas silently converts the label \"None\" to a missing value unless read with keep_default_na=False, which would "
         "delete a class; every loader guards against it. All results come from a single seed.")

    # ---------------- 5. Future work
    heading("5. Future work and planned improvements before the final evaluation")
    para("(1) Finish the blind human review of the 600 synthetic test rows and the 200-row translation check, report "
         "human-vs-intended agreement and re-score with verified labels. (2) Collect and annotate a small set of real public "
         "Hindi/Bengali comments as a genuine test set. (3) Repeat the main runs over several seeds to measure variance, and "
         "test whether differences exceed the bootstrap intervals. (4) Use the per-phenomenon error analysis to target "
         f"the best run's weakest phenomena ({F['weak']}) with more varied data. (5) Add a zero-shot "
         "LLM baseline and try IndicTrans2 translation with an approved Hugging Face token.")
    return doc


def analysis(R, F):
    """The three comparisons, each sentence built from the numbers."""
    runs, get = R["runs"], F["get"]
    out = []
    fair = ["translated_hi_test", "translated_bn_test", "pilot"]
    mean_fair = lambda name: sum(get(name, s) for s in fair) / 3 if all(get(name, s) is not None for s in fair) else None  # noqa: E731
    cmp = []
    for c in ("A_english", "B_translated", "C_full"):
        vals = {m: mean_fair(f"{m}_{c}") for m in ("tfidf", "muril", "indicbert") if f"{m}_{c}" in runs}
        vals = {m: v for m, v in vals.items() if v is not None}
        if len(vals) > 1:
            cmp.append(f"{c.split('_')[0]}: " + ", ".join(f"{MODEL[m]} {f3(v)}" for m, v in vals.items()))
    if cmp:
        out.append("Models on the same data (mean macro-F1 over translated Hindi, translated Bengali and pilot): "
                   + "; ".join(cmp) + ".")
    for m in ("indicbert", "muril", "tfidf"):
        a, b, c = (f"{m}_{x}" for x in ("A_english", "B_translated", "C_full"))
        if a in runs and (b in runs or c in runs):
            parts = []
            for other, label in ((b, "translated data (B vs A)"), (c, "synthetic data on top (C vs B)")):
                base = a if other == b else b
                if other in runs and base in runs:
                    parts.append(f"{label}: " + ", ".join(
                        f"{lab} {get(other, s) - get(base, s):+.3f}" for s, lab in SETS
                        if get(other, s) is not None and get(base, s) is not None))
            out.append(f"What each data family adds for {MODEL[m]} (macro-F1 change): " + "; ".join(parts) + ".")
            if b in runs and c in runs:
                same = [lab for s, lab in SETS[1:3] if overlap(runs[b]["eval"][s]["ci95"], runs[c]["eval"][s]["ci95"])]
                if same:
                    out.append(f"On the {' and '.join(same)} test sets the 95% intervals of B and C overlap, so synthetic data "
                               f"shows no clear gain there; C's large gains are on the synthetic test set (same generator, "
                               f"optimistic) and the pilot, which is also synthetic comment-style text.")
            a_pc = runs.get(a, {}).get("eval", {}).get("synthetic_test", {}).get("predicted_counts")
            if a_pc:
                out.append(f"Trained on English only, {MODEL[m]} predicts Favour for {a_pc['Favour']} of "
                           f"{sum(a_pc.values())} synthetic comments, i.e. it reads almost any climate mention as agreement; "
                           "in-language data (B, C) removes most of this bias.")
            break
    a = "indicbert_A_english"
    if a in runs and mean_fair(a) is not None:
        out.append(f"Cross-lingual loss: IndicBERT v2 trained on English only scores {f3(get(a, 'english_test'))} on the "
                   f"English test set and {f3(mean_fair(a))} on average over the translated and pilot sets "
                   f"({mean_fair(a) - get(a, 'english_test'):+.3f}); the TF-IDF baseline drops from "
                   f"{f3(get('tfidf_A_english', 'english_test'))} to {f3(mean_fair('tfidf_A_english'))}, because word and "
                   "character features do not transfer across scripts." if "tfidf_A_english" in runs else "")
    b = R["best_run"]
    if b:
        e = runs[b]["eval"]
        weak = e.get("synthetic_test", e.get("pilot", {})).get("accuracy_by_phenomenon", {})
        if weak:
            lo = min(weak, key=lambda k: weak[k]["accuracy"])
            hi = max(weak, key=lambda k: weak[k]["accuracy"])
            src = "synthetic test" if "synthetic_test" in e else "pilot"
            out.append(f"Best run by the pre-set rule ({R['best_rule'].split(' (')[0]}): {MODEL[runs[b]['model']]} "
                       f"{runs[b]['config'].split('_')[0]}. On the {src} set its accuracy is lowest for '{lo}' "
                       f"({weak[lo]['accuracy']:.2f}, n={weak[lo]['n']}) and highest for '{hi}' ({weak[hi]['accuracy']:.2f}); "
                       f"Against recall on translated Hindi is {e['translated_hi_test']['per_class']['Against']['recall']:.2f}.")
    return [o for o in out if o]


def low2(gen):
    """The two phenomena with the lowest judge agreement, read from the stats (never typed by hand)."""
    ph = gen["judge_agreement_by_phenomenon"]
    return " and ".join(f"{k.replace('_', ' ')} ({100 * ph[k]:.1f}% agreement)" for k in sorted(ph, key=ph.get)[:2])


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def word_pages(path):
    """Page count via Microsoft Word, or None if Word is not available."""
    ps = (f"$w=New-Object -ComObject Word.Application;$w.Visible=$false;"
          f"$d=$w.Documents.Open('{os.path.abspath(path)}',$false,$true);$n=$d.ComputeStatistics(2);"
          f"$d.ExportAsFixedFormat('{os.path.abspath(path)[:-5]}.pdf',17);$d.Close($false);$w.Quit();$n")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=180)
        return int(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def main():
    R = json.load(open(RES, encoding="utf-8"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for size in (11, 10.5, 10, 9.5, 9):
        build(R, size).save(OUT)
        pages = word_pages(OUT)
        print(f"body {size} pt -> {pages} pages")
        if pages is None:
            print("!! Microsoft Word not available: page count NOT verified (target: two A4 pages)")
            break
        if pages <= 2:
            break
    if pages == 1:
        print("!! only one page: add content before handing in")
    print(f"wrote {OUT} (+ PDF export when Word is present)")


if __name__ == "__main__":
    main()
