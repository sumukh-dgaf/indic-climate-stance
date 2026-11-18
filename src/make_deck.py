"""Build outputs/Milestone1_Sumukh_S24CSEU0570.pptx (10 slides) from the generated JSON/PNG outputs.

Numbers come from outputs/eda_stats.json and outputs/baseline_results.json; examples come from the pilot CSV by id.
Run: python src/make_deck.py   (after eda.py and baseline.py)
"""
import json
import math
import os

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

OUT = "outputs/"
INK, MUTED = RGBColor(0x1B, 0x26, 0x3B), RGBColor(0x5C, 0x67, 0x7D)
NAVY, TEAL, CORAL, GREY = RGBColor(0x1F, 0x2A, 0x44), RGBColor(0x2A, 0x9D, 0x8F), RGBColor(0xE7, 0x6F, 0x51), RGBColor(0x8D, 0x99, 0xAE)
PANEL, WHITE, AMBER = RGBColor(0xF1, 0xF4, 0xF8), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xE9, 0xA8, 0x20)
LABEL_COLOR = {"Favour": TEAL, "Against": CORAL, "None": GREY}
WARN = []

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def _est_lines(text, width_in, size):
    wide = any(0x0900 <= ord(c) <= 0x09FF for c in text)  # Indic glyphs are wider than Latin
    cpl = max(1, int(width_in * 72 / (size * (0.62 if wide else 0.5))))
    return max(1, sum(math.ceil(max(1, len(p)) / cpl) for p in text.split("\n")))


def _style(run, size, bold, color, italic=False):
    f = run.font
    f.size, f.bold, f.italic = Pt(size), bold, italic
    f.color.rgb = color
    f.name = "Calibri"
    rpr = run._r.get_or_add_rPr()
    cs = etree.SubElement(rpr, qn("a:cs"))
    cs.set("typeface", "Nirmala UI")  # complex-script font so Devanagari/Bengali render on Windows lab PCs


def text(slide, x, y, w, h, paras, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         fill=None, line=None, pad=0.1, space=4, shape=MSO_SHAPE.RECTANGLE):
    """paras: list of str | list[(text, {size,bold,color,italic})] runs. Adds a shape with a text frame."""
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.5)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(pad)
    tf.margin_top = tf.margin_bottom = Inches(0.05)
    need = 0.0
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space)
        runs = [(p, {})] if isinstance(p, str) else p
        plain, psize = "", size
        for t, o in runs:
            r = para.add_run()
            r.text = t
            psize = o.get("size", size)
            _style(r, psize, o.get("bold", bold), o.get("color", color), o.get("italic", False))
            plain += t
        need += _est_lines(plain, w - 2 * pad, psize) * psize * 1.22 / 72 + space / 72
    if need > h:
        WARN.append(f"slide {len(prs.slides)}: text may overflow ({need:.2f}in > {h:.2f}in): {paras[0] if isinstance(paras[0], str) else paras[0][0][0]!r:.50}")
    return sh


def image(slide, path, x, y, w=None, h=None):
    iw, ih = Image.open(path).size
    if w is not None:
        h = w * ih / iw
    else:
        w = h * iw / ih
    slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def table(slide, x, y, w, col_w, rows, size=15, header_fill=NAVY, row_h=0.5):
    nrows, ncols = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(row_h * nrows))
    tbl = shp.table
    tbl.horz_banding = False
    total = sum(col_w)
    heights = 0.0
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(w * cw / total)
    for i, row in enumerate(rows):
        worst = 1
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = header_fill if i == 0 else (WHITE if i % 2 else PANEL)
            tf = c.text_frame
            tf.word_wrap = True
            r = tf.paragraphs[0].add_run()
            r.text = str(val)
            _style(r, size, i == 0 or j == 0, WHITE if i == 0 else INK)
            worst = max(worst, _est_lines(str(val), w * col_w[j] / total - 0.16, size))
        rh = worst * size * 1.22 / 72 + 0.12
        tbl.rows[i].height = Inches(max(rh, 0.38))
        heights += max(rh, 0.38)
    if y + heights > 6.85:
        WARN.append(f"slide {len(prs.slides)}: table may run past the footer (ends {y + heights:.2f}in)")
    return heights


FOOTER, TOTAL = "Milestone 1", 10  # rebound for the Round 2 deck at the end of this file


def new_slide(num, tag, title, notes=""):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.55, 0.28, 9.5, 0.4, [tag.upper()], size=14, color=TEAL, bold=True, pad=0)
    text(s, 0.55, 0.62, 12.2, 0.95, [title], size=30, color=NAVY, bold=True, pad=0, anchor=MSO_ANCHOR.TOP)
    text(s, 0.55, 7.05, 10, 0.3, [f"Sumukh Gupta, Neya Tripathi, Aaditya Goel, Kushika Mishra, Eklavya Sharma  |  CSET346 NLP  |  {FOOTER}"], size=11, color=MUTED, pad=0)
    text(s, 11.6, 7.05, 1.2, 0.3, [f"{num} / {TOTAL}"], size=11, color=MUTED, pad=0, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def pct(x):
    return f"{x:.1f}%"


# ------------------------------------------------------------------ data
E = json.load(open(OUT + "eda_stats.json"))
B = json.load(open(OUT + "baseline_results.json"))
P = pd.read_csv("data/pilot_indic_stance.csv", dtype=str, keep_default_na=False).set_index("id")
G = {(r["features"], r["model"]): r for r in B["A_english_cv"]}
best = G[("word+char", "LogReg")]
floor = G[("-", "Dummy(majority)")]
abl = B["A_class_weight_ablation"]
D = B["D_transfer_to_pilot"]
eng, pil, X = E["english_merged"], E["pilot"], E["extra"]
ex = lambda i: P.loc[i, "comment"]  # noqa: E731

# ------------------------------------------------------------------ 1 title
s = prs.slides.add_slide(BLANK)
text(s, 0, 0, 13.333, 7.5, [""], fill=NAVY)
text(s, 0.8, 0.7, 11, 0.5, ["CSET346  NATURAL LANGUAGE PROCESSING  |  MILESTONE 1"], size=16, color=TEAL, bold=True, pad=0)
text(s, 0.8, 1.5, 11.8, 2.5, ["Detecting Stance in Media on Global Warming for Indic Languages"], size=46, color=WHITE, bold=True, pad=0)
text(s, 0.8, 3.9, 11.5, 0.9, ["Favour / Against / None toward one claim, in Hindi and Bengali comments"], size=24, color=RGBColor(0xC9, 0xD3, 0xE3), pad=0)
text(s, 0.8, 4.85, 3.6, 2.2, ["Sumukh Gupta  |  S24CSEU0570", "Neya Tripathi  |  S24CSEU0609", "Aaditya Goel  |  S24CSEU0614", "Kushika Mishra  |  S24CSEU0652", "Eklavya Sharma  |  S24CSEU0798"], size=17, color=WHITE, pad=0)
text(s, 4.7, 4.85, 3.7, 2.2, ["B.Tech CSE, Bennett University", "Instructor: Dr. Supriya Chanda"], size=17, color=RGBColor(0xC9, 0xD3, 0xE3), pad=0)
text(s, 8.6, 5.2, 4.2, 1.4, ["Derived from the AISoMe 2026 shared task at FIRE 2026"], size=16, color=RGBColor(0xC9, 0xD3, 0xE3), pad=0)
s.notes_slide.notes_text_frame.text = "Introduce: project, claim, that the task comes from AISoMe 2026 (English train, Hindi/Bengali test)."

# ------------------------------------------------------------------ 2 problem identification
s = new_slide(2, "1 | Problem identification", "One claim, three labels, Hindi and Bengali comments",
              "The claim never appears in the comments. Examples are from our team-authored pilot corpus, not scraped data.")
text(s, 0.55, 1.5, 12.2, 0.8, ["“Climate change and global warming is a serious concern.”"], size=26, color=NAVY, bold=True,
     fill=PANEL, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
cols = [("Favour", "Agrees it is a serious concern", "P003", "P078"),
        ("Against", "Disagrees, minimises, denies", "P027", "P102"),
        ("None", "Neutral, off-topic or unclear", "P057", "P132")]
for k, (lab, desc, hi, bn) in enumerate(cols):
    x = 0.55 + k * 4.12
    text(s, x, 2.55, 3.95, 0.6, [lab], size=26, bold=True, color=WHITE, fill=LABEL_COLOR[lab], anchor=MSO_ANCHOR.MIDDLE,
         align=PP_ALIGN.CENTER)
    text(s, x, 3.2, 3.95, 0.5, [desc], size=16, color=MUTED, align=PP_ALIGN.CENTER)
    text(s, x, 3.75, 3.95, 1.45, [[("Hindi  ", {"bold": True, "size": 14, "color": MUTED})], ex(hi)], size=19, fill=PANEL)
    text(s, x, 5.3, 3.95, 1.45, [[("Bengali  ", {"bold": True, "size": 14, "color": MUTED})], ex(bn)], size=19, fill=PANEL)
text(s, 0.55, 6.72, 12.2, 0.3, ["Examples: team-authored pilot corpus (synthetic, development only). Metric: macro-F1."], size=12, color=MUTED, pad=0)

# ------------------------------------------------------------------ 3 problem analysis
s = new_slide(3, "1 | Problem analysis", "Why this is not simple text classification",
              "Four arguments: external claim, shared vocabulary, implicit stance, heterogeneous None. Stats are measured.")
cards = [
    ("1  The claim is not in the text", "Against, because the claim is about warming:", ex("P046") + "  (Against)", None),
    ("2  Same words, opposite stance", f"{X['favour_against_vocab_overlap_pct']}% of Against vocabulary also appears in Favour (English set).",
     ex("P010") + "  (Favour)\n" + ex("P027") + "  (Against)", None),
    ("3  Stance is often implicit", "World knowledge needed; no climate word:", ex("P019") + "  (Favour)", None),
    ("4  None is a junk drawer", f"Neutral + off-topic + ambiguous. {X['raw_ties']} of {X['raw_total']:,} annotated sentences "
     f"({pct(X['raw_ties_pct'])}) had no annotator majority.", ex("P051") + "  (None)", None),
]
for k, (h, body, exm, _) in enumerate(cards):
    x, y = 0.55 + (k % 2) * 6.15, 1.55 + (k // 2) * 2.7
    text(s, x, y, 6.0, 0.5, [h], size=21, bold=True, color=NAVY, pad=0.12)
    text(s, x, y + 0.5, 6.0, 0.7, [body], size=16, color=MUTED, pad=0.12)
    text(s, x, y + 1.2, 6.0, 1.35, [exm], size=15, fill=PANEL, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.55, 6.8, 12.2, 0.25, ["Plus: English training data, Hindi/Bengali test data. Examples: pilot corpus (synthetic)."], size=11, color=MUTED, pad=0)

# ------------------------------------------------------------------ 4 existing systems
s = new_slide(4, "2 | Review of existing systems", "What exists, what it reports, why it falls short",
              "Numbers are as reported by each source on its own data; not comparable with each other or with ours.")
rows = [["Approach", "Representative work", "Reported result", "Limitation for us"],
        ["Tweet stance, 3 labels", "SemEval-2016 Task 6 (Mohammad)", "see paper", "English; per-target"],
        ["News-sentence stance", "GWSD (Luo et al. 2020)", "see paper", "English news, not comments"],
        ["Climate tweets, 6 layers", "ClimaConvo (Shiwakoti 2024)", "15,309 rows", "IDs only: no text"],
        ["Multimodal video", "MultiClimate (Wang 2024)", "acc 0.747, F1 0.749", "English transcripts"],
        ["Bangla news stance", "Dhoroni (Wasi 2024)", "acc 0.635, F1 0.308", "News; minority classes weak"],
        ["Code-mixed Hindi", "Swami et al. 2018", "58.7% accuracy", "Target: demonetisation"],
        ["Same task, translate-train", "Squad4, AISoMe 2026", "F1 0.42-0.53 (val 0.65-0.68)", "Large val-to-test gap"]]
table(s, 0.55, 1.55, 12.2, [2.5, 3.0, 2.8, 2.7], rows, size=16, row_h=0.6)
text(s, 0.55, 6.62, 12.2, 0.4, ["Squad4: a competing team; only their reported scores are cited. Ours (measured): TF-IDF macro-F1 "
                                f"{best['macro_f1']:.3f} on English CV."], size=13, color=MUTED, pad=0)

# ------------------------------------------------------------------ 5 research gap
s = new_slide(5, "2 | Research gap", "No resource matches target, language and domain",
              "Headline measured result: English-trained TF-IDF on the Indic pilot gives macro-F1 0.341 and predicts None for 131/150. "
              "Gap table: data/DATASET_INVENTORY.md. Pilot is synthetic, development only.")
text(s, 0.55, 1.55, 3.6, 0.9, ["English labelled data"], size=22, bold=True, color=WHITE, fill=TEAL, align=PP_ALIGN.CENTER,
     anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.55, 2.5, 3.6, 0.6, ["GWSD news + SemEval tweets"], size=16, color=MUTED, align=PP_ALIGN.CENTER)
text(s, 1.75, 3.05, 1.2, 0.75, ["↓"], size=34, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
text(s, 0.55, 3.85, 3.6, 0.9, ["Hindi / Bengali comments"], size=22, bold=True, color=WHITE, fill=CORAL, align=PP_ALIGN.CENTER,
     anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.55, 4.8, 3.6, 0.9, ["hidden test: code-mixing, spelling variation, negation, implicit stance"], size=14, color=MUTED,
     align=PP_ALIGN.CENTER)
rows = [["Resource", "Our claim", "Hindi / Bengali", "Comments"],
        ["SemEval-2016 climate", "Yes", "No", "Yes"],
        ["GWSD", "Yes", "No", "No (news)"],
        ["ClimaConvo", "Close", "No", "Yes, but no text"],
        ["MultiClimate", "Close", "No", "No (transcripts)"],
        ["Dhoroni", "Close", "Bengali", "No (news)"],
        ["Swami 2018", "No (demonetisation)", "Romanised Hindi", "Yes"],
        ["AISoMe test (hidden)", "Yes", "Yes", "Yes"]]
table(s, 4.6, 1.55, 8.15, [2.6, 2.2, 2.0, 2.2], rows, size=15, row_h=0.5)
text(s, 0.55, 5.85, 12.2, 1.1, [[("Measured, not assumed: ", {"bold": True, "color": CORAL}),
     (f"English-trained TF-IDF on the Indic pilot reaches macro-F1 {D['macro_f1']:.3f} (floor 0.167) and predicts None for "
      f"{D['predicted_class_counts']['None']} of 150: cross-lingual transfer collapses to majority-class prediction, so "
      f"in-language data and Indic-specific models are needed. {X['favour_against_vocab_overlap_pct']}% of Against vocabulary "
      "also appears in Favour: not simple text classification. ", {}),
     ("(Pilot: synthetic, development only.)", {"italic": True, "color": MUTED})]],
     size=15, fill=RGBColor(0xFD, 0xEC, 0xE8), line=CORAL, pad=0.15, anchor=MSO_ANCHOR.MIDDLE)

# ------------------------------------------------------------------ 6 dataset exploration
s = new_slide(6, "3 | Dataset exploration and statistics", "What data we actually have",
              "Usable dataset: PARTIALLY. English target-correct data yes; Hindi/Bengali comments only the synthetic pilot.")
image(s, OUT + "fig_class_distribution.png", 0.45, 1.55, w=7.0)
text(s, 7.7, 1.55, 5.1, 0.55, ["Usable dataset found: PARTIALLY"], size=22, bold=True, color=WHITE, fill=AMBER,
     anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
text(s, 7.7, 2.25, 5.1, 4.6, [
    [("English merged: ", {"bold": True}), (f"{eng['n']:,} labelled (GWSD news + SemEval tweets)", {})],
    [("Against is rare: ", {"bold": True}), (f"{eng['class_pct']['Against']}% (SemEval: 26 of 564)", {})],
    [("Hapax rate: ", {"bold": True}), (f"{eng['hapax_rate_pct']}% English, {pil['hapax_rate_pct']}% pilot", {})],
    [("Majority floor: ", {"bold": True}), (f"macro-F1 {eng['majority_floor_macro_f1']:.3f}", {})],
    [("No annotator majority: ", {"bold": True}), (f"{X['raw_ties']} of {X['raw_total']:,} GWSD ({pct(X['raw_ties_pct'])}); "
                                                f"{X['ties']} remain after dedup, dropped", {})],
    [("Median length: ", {"bold": True}), (f"{eng['words']['median']:.0f} words (pilot {pil['words']['median']:.0f})", {})],
    [("Hindi/Bengali: ", {"bold": True, "color": CORAL}), ("150 team-authored synthetic comments, development only", {})],
], size=18, space=7)
text(s, 0.45, 5.3, 7.0, 1.4, ["Not obtained: official AISoMe data; ClimaConvo text (IDs only). Dhoroni (Bengali news) and Swami "
                              "(demonetisation) kept out of the training pool."], size=15, color=MUTED, pad=0.1)

# ------------------------------------------------------------------ 7 objectives and methodology
s = new_slide(7, "3 | Objectives and methodology", "Pipeline: clean text to cross-lingual evaluation",
              "Green = implemented and run today. Amber = written, not yet run. Grey = planned for Milestone 2.")
text(s, 0.55, 1.5, 12.2, 1.2, [
    "1  Understand and measure the task and data honestly",
    "2  Build a reproducible baseline with a floor to beat",
    "3  Design a target-aware multilingual model and a cross-lingual strategy"], size=19, color=INK, space=3)
steps = [("Preprocess", "NFC, noise strip, script ID", TEAL, "done"),
         ("Baselines", "TF-IDF word + char, majority floor", TEAL, "done"),
         ("Target-aware encoder", "MuRIL on claim [SEP] comment", AMBER, "written"),
         ("Cross-lingual", "zero-shot vs translate-train", GREY, "plan"),
         ("Evaluate", "macro-F1, per phenomenon", GREY, "plan")]
for k, (h, sub, col, st) in enumerate(steps):
    x = 0.55 + k * 2.5
    text(s, x, 3.2, 2.2, 0.95, [h], size=20, bold=True, color=WHITE, fill=col, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x, 4.25, 2.2, 1.0, [sub], size=16, color=INK, align=PP_ALIGN.CENTER)
    text(s, x, 5.25, 2.2, 0.4, [st], size=14, color=col, bold=True, align=PP_ALIGN.CENTER)
    if k < 4:
        text(s, x + 2.17, 3.35, 0.4, 0.6, ["→"], size=28, color=NAVY, bold=True, pad=0, align=PP_ALIGN.CENTER)
text(s, 0.55, 5.9, 12.2, 0.9, ["Models are compared against the floor; the pilot is for development and error analysis only."], size=16, color=MUTED, pad=0)

# ------------------------------------------------------------------ 8 algorithms
s = new_slide(8, "4 | Relevance of the techniques", "Why each choice, with evidence",
              "Char n-grams: hapax rate. MuRIL: IndicXNLI. Pair input: stance is relative to a claim. Macro-F1 + weights: rare Against.")
cards = [("Char n-grams (2-5)", f"Hapax rate {eng['hapax_rate_pct']}% English, {pil['hapax_rate_pct']}% pilot: most words seen once. "
          "Sub-strings are shared across spellings and Romanised text."),
         ("MuRIL encoder", "IndicXNLI: MuRIL beats XLM-R, mBERT and IndicBERT on Indic languages; English + Indic training is best. "
          "Pretrained with transliterated text."),
         ("claim [SEP] comment", "Stance is defined relative to the claim, so the model must see both, not just the topic words."),
         ("Macro-F1 + class weights", f"Floor {floor['macro_f1']:.2f}. Weights: macro-F1 {abl['none']['macro_f1']:.2f} to "
          f"{abl['balanced']['macro_f1']:.2f}, Against recall {abl['none']['against_recall']:.2f} to {abl['balanced']['against_recall']:.2f}.")]
for k, (h, body) in enumerate(cards):
    x, y = 0.55 + (k % 2) * 6.15, 1.55 + (k // 2) * 2.7
    text(s, x, y, 6.0, 0.6, [h], size=24, bold=True, color=WHITE, fill=[TEAL, NAVY, NAVY, TEAL][k], anchor=MSO_ANCHOR.MIDDLE)
    text(s, x, y + 0.65, 6.0, 1.85, [body], size=22, fill=PANEL, pad=0.15)

# ------------------------------------------------------------------ 9 results
s = new_slide(9, "Initial implementation and preliminary results", "What runs today",
              "Headline: English-to-Indic zero-shot collapses to majority-class prediction (pilot: synthetic, dev only). "
              "English CV numbers are optimistic (no document ids, one hashtag cluster).")
image(s, OUT + "fig_baseline_comparison.png", 0.45, 1.45, w=5.9)
image(s, OUT + "fig_confusion_english_cv.png", 6.5, 1.45, w=3.2)
le = B["A_best"]["largest_error"]
text(s, 9.9, 1.6, 3.0, 2.4, ["Best (English CV):", "word+char LogReg", f"macro-F1 {best['macro_f1']:.3f}", f"floor {floor['macro_f1']:.3f}"],
     size=17, color=NAVY, bold=True, pad=0.05)
text(s, 0.55, 4.8, 12.2, 1.3, [[("Headline finding: ", {"bold": True, "color": CORAL}),
     (f"English to Indic zero-shot collapses: macro-F1 {D['macro_f1']:.3f} vs floor 0.167, predicting None for "
      f"{D['predicted_class_counts']['None']} of 150 comments. With {X['favour_against_vocab_overlap_pct']}% of Against vocabulary "
      "also in Favour, this is not simple text classification. ", {}),
     ("(Pilot: synthetic, dev only; not comparable to official results.)", {"italic": True, "color": MUTED})]],
     size=17, fill=RGBColor(0xFD, 0xEC, 0xE8), line=CORAL, pad=0.15, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.55, 6.2, 12.2, 0.8, [
    [("Runs today: ", {"bold": True}), ("loader + validation gate, preprocessing, EDA, TF-IDF baselines, self-checks.", {})],
    [("Error analysis: ", {"bold": True}), (f"largest confusion is true {le['true']} predicted {le['pred']} ({le['n']}): factual "
                                           "sentences with no opinion marker. Against recall "
                                           f"{B['A_best']['against_recall']:.2f}.", {})]], size=14, space=2)

# ------------------------------------------------------------------ 10 feasibility
s = new_slide(10, "2 | Feasibility, risks and next steps", "Feasible, with known risks",
              "Biggest risk: validation-to-test gap reported by a peer team (0.65-0.68 to 0.42-0.53).")
cols = [("Compute", ["Baselines run on a laptop CPU in minutes (done)",
                     "Fine-tuning a base-size encoder on short texts: Colab GPU planned",
                     "Runtime: not yet measured"]),
        ("Risks and mitigation", ["Official data not in hand: obtain from the course email; pilot for development only",
                                       "Against is rare: class weights, macro-F1",
                                       "Validation-to-test gap: per-phenomenon evaluation"]),
        ("Milestone 2", ["Run MuRIL on claim [SEP] comment (XLM-R as comparison)",
                         "Zero-shot vs translate-train vs both",
                         "Zero-shot LLM baseline",
                         "Swap in official data"])]
for k, (h, items) in enumerate(cols):
    x = 0.55 + k * 4.12
    text(s, x, 1.6, 3.95, 0.6, [h], size=22, bold=True, color=WHITE, fill=[TEAL, CORAL, NAVY][k], anchor=MSO_ANCHOR.MIDDLE)
    text(s, x, 2.3, 3.95, 4.3, items, size=20, fill=PANEL, pad=0.15, space=12)

# ------------------------------------------------------------------ save
os.makedirs(OUT, exist_ok=True)
path = OUT + "Milestone1_Sumukh_S24CSEU0570.pptx"
assert len(prs.slides) <= 10
prs.save(path)
print(f"saved {path} with {len(prs.slides)} slides")
for w in WARN:
    print("WARNING", w)
print("fit warnings:", len(WARN))


# ================================================================== Round 2 deck (every number from round2_results.json)
def class_chart(ds, path):
    """Stacked 100% bars: class distribution of each labelled data family (counts written on the bars)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fam = [("English, human-labelled", {k: sum(ds["english"][s][k] for s in ds["english"]) for k in ("Favour", "Against", "None")})]
    for lang, name in (("hi", "Hindi, synthetic"), ("bn", "Bengali, synthetic")):
        if lang in ds.get("synthetic_by_language", {}):
            fam.append((name, ds["synthetic_by_language"][lang]))
    fam.append(("Pilot (hi+bn)", ds["pilot"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    for r, (name, d) in enumerate(reversed(fam)):
        tot, left = sum(d[k] for k in ("Favour", "Against", "None")), 0
        for k in ("Favour", "Against", "None"):
            w = d[k] / tot
            ax.barh(r, w, left=left, color="#" + str(LABEL_COLOR[k]), edgecolor="white", height=0.62)
            ax.text(left + w / 2, r, f"{d[k]:,}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            left += w
    ax.set_yticks(range(len(fam)), [n for n, _ in reversed(fam)], fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", labelsize=8.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[matplotlib.patches.Patch(color="#" + str(LABEL_COLOR[k]), label=k) for k in ("Favour", "Against", "None")],
              ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def round2_deck():
    """Final 10-slide Round 2 deck. Every number comes from outputs/round2_results.json (written by evaluate.py)."""
    global prs, BLANK, FOOTER, TOTAL
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_report import MODEL, SETS, facts, overlap
    R = json.load(open(OUT + "round2_results.json", encoding="utf-8"))
    F = facts(R)
    runs, ds, get, g = R["runs"], R["dataset"], F["get"], F["gen"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    BLANK = prs.slide_layouts[6]
    FOOTER, TOTAL = "Round 2", 10
    WARN.clear()
    star = "" if F["verified"] else "*"
    unverified = "" if F["verified"] else "*Synthetic test labels not yet human-verified."
    LIGHT = RGBColor(0xC9, 0xD3, 0xE3)

    # 1 title
    s = prs.slides.add_slide(BLANK)
    text(s, 0, 0, 13.333, 7.5, [""], fill=NAVY)
    text(s, 0.8, 0.7, 11, 0.5, ["CSET346  NATURAL LANGUAGE PROCESSING  |  ROUND 2"], size=16, color=TEAL, bold=True, pad=0)
    text(s, 0.8, 1.45, 11.8, 2.2, ["Stance Detection on Climate-Change Comments in Hindi and Bengali"], size=42, color=WHITE,
         bold=True, pad=0)
    text(s, 0.8, 3.65, 11.5, 0.9, ["Fine-tuned IndicBERT v2, cross-lingual transfer from English, and a new Hindi-Bengali dataset"],
         size=21, color=LIGHT, pad=0)
    text(s, 0.8, 4.75, 4.6, 2.2, ["Sumukh Gupta  |  S24CSEU0570", "Neya Tripathi  |  S24CSEU0609", "Aaditya Goel  |  S24CSEU0614",
                                  "Kushika Mishra  |  S24CSEU0652", "Eklavya Sharma  |  S24CSEU0798"], size=17, color=WHITE, pad=0)
    text(s, 5.8, 4.75, 6.8, 2.2, ["B.Tech CSE, Bennett University", "Instructor: Dr. Supriya Chanda",
                                  "github.com/sumukh-dgaf/indic-climate-stance"], size=17, color=LIGHT, pad=0)

    # 2 task
    s = new_slide(2, "The task", "Is a comment for or against the claim?",
                  "Examples are illustrative, written in Romanised Hindi so they display on any laptop.")
    text(s, 0.55, 1.6, 12.2, 0.8, ["Claim:  “Climate change and global warming is a serious concern.”"], size=22, bold=True,
         color=WHITE, fill=NAVY, anchor=MSO_ANCHOR.MIDDLE, pad=0.25)
    cards = [("Favour", "agrees it is serious", "“Is baar garmi ne saare record tod diye, ab kuch karna padega”"),
             ("Against", "denies or dismisses it", "“Global warming sab propaganda hai, thand to har saal padti hai”"),
             ("None", "no stance on the claim", "“Video ka link bhejo bhai”")]
    for k, (lab, mean, ex) in enumerate(cards):
        x = 0.55 + k * 4.12
        text(s, x, 2.65, 3.95, 0.55, [lab], size=20, bold=True, color=WHITE, fill=LABEL_COLOR[lab], anchor=MSO_ANCHOR.MIDDLE, pad=0.15)
        text(s, x, 3.2, 3.95, 1.55, [[(mean, {"bold": True})], [(ex, {"italic": True})]], size=16, fill=PANEL, pad=0.15, space=4)
    text(s, 0.55, 4.95, 7.6, 1.9, [[("Why it is hard", {"bold": True, "color": NAVY})], "Stance is often implicit, sarcastic or negated",
                                   "Code-mixed and Romanised spelling varies freely",
                                   "No labelled Hindi/Bengali data for this claim exists"], size=18, pad=0, space=5)
    text(s, 8.45, 4.95, 4.3, 1.9, [[("Metric: macro-F1", {"bold": True})],
                                   "F1 of the three classes averaged equally, so the rare Against class counts as much as the others"],
         size=16, fill=PANEL, pad=0.18, space=6)

    # 3 dataset
    s = new_slide(3, "1 | Dataset", "Four data families, each labelled by provenance",
                  "No official data was provided. None of these are real Hindi/Bengali YouTube comments; say so.")
    rows = [["Data family", "Train", "Val", "Test"]]
    rows.append(["English, human-labelled (GWStance + SemEval-2016)"] + [f"{F['eng'][k]:,}" for k in ("train", "val", "test")])
    for lang, name in (("hi", "Hindi"), ("bn", "Bengali")):
        if f"{lang}_train" in F["tr_sizes"]:
            rows.append([f"{name}, machine-translated (NLLB-200)"] + [f"{F['tr_sizes'][f'{lang}_{k}']['total']:,}" for k in ("train", "val", "test")])
    if F["syn"]:
        rows.append(["Hindi + Bengali, synthetic (Gemini)"] + [f"{F['syn'][k]:,}" for k in ("train", "val", "test")])
    rows.append(["Pilot, team-authored synthetic", "-", "-", str(F["pilot"])])
    table(s, 0.55, 1.65, 6.3, [3.6, 0.9, 0.9, 0.9], rows, size=13, row_h=0.55)
    chart = OUT + "fig_r2_class_distribution.png"
    class_chart(ds, chart)
    image(s, chart, 7.05, 1.55, w=5.75)
    text(s, 0.55, 5.35, 12.2, 1.5, [f"Class distribution: Against is the rare class in English ({F['eng_against_test']} of "
                                    f"{F['eng']['test']} test rows); the synthetic data was generated balanced by design.",
                                    "Translated sets copy the English labels and class distribution." + ("  " + unverified if unverified else "")],
         size=15, color=MUTED, pad=0, space=4)

    # 4 data building
    s = new_slide(4, "1 | Dataset", "How the Hindi and Bengali data was built and checked",
                  "Split first, then translate: a sentence and its translation never sit in different splits.")
    flow = [f"{g['generated']:,} generated" if g else "generated", f"-{g['removed_script']:,} wrong script" if g else "script check",
            f"-{g['removed_exact_duplicate'] + g['removed_near_duplicate']:,} duplicates" if g else "duplicates",
            f"-{g['removed_judge_disagrees']:,} judge disagrees" if g else "judge", f"{g['kept_after_quota']:,} kept" if g else "kept"]
    for k, st in enumerate(flow):
        text(s, 0.55 + k * 2.47, 1.65, 2.3, 0.9, [st], size=16, bold=True, color=WHITE, fill=[TEAL, NAVY][k % 2],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    left = [[("Synthetic comments (Gemini)", {"bold": True, "color": NAVY})], "Generated per language, label and phenomenon: code-mixed, Romanised, sarcasm, negation, "
            "implicit, emoji, off-topic", "A second Gemini model judges each comment without seeing the intended label"]
    if g:
        left.append(f"Judge agreement {100 * g['judge_agreement_overall']:.1f}% overall, only "
                    f"{100 * g['judge_agreement_by_phenomenon']['sarcasm']:.1f}% on sarcasm")
    text(s, 0.55, 2.85, 6.0, 3.9, left, size=17, pad=0, space=8)
    rv, tr = ds.get("synthetic_review"), ds.get("translation_review")
    right = [[("Human checks by the team", {"bold": True, "color": NAVY})]]
    right.append(f"Blind review of the {rv['reviewed']} synthetic test comments: agreed with the intended label "
                 f"{100 * rv['agreement_with_intended']:.1f}%, {rv['labels_changed']} labels corrected" if rv else
                 "Blind review of all 600 synthetic test comments: in progress (labels not yet human-verified)")
    if tr:
        right.append("Translation check, 100 rows per language: stance kept in " + ", ".join(
            f"{100 * tr[l]['label_survives_rate']:.0f}% ({'Hindi' if l == 'hi' else 'Bengali'})" for l in ("hi", "bn")
            if tr.get(l, {}).get("label_survives_rate") is not None))
    right.append("English and translated text is rebuilt by script, never committed (GWStance has no licence)")
    text(s, 6.85, 2.85, 5.9, 3.9, right, size=17, fill=PANEL, pad=0.2, space=8)

    # 5 methodology
    s = new_slide(5, "2 | Methodology", "Pair input, three data configurations, three models",
                  "Claim stays English. Each configuration selects its best epoch on its own validation data; tests never used to select.")
    steps = ["English splits", "Translate (NLLB)", "Synthetic + judge", "Train A / B / C", "Predict 5 test sets", "Evaluate"]
    for k, st in enumerate(steps):
        text(s, 0.55 + k * 2.08, 1.65, 1.9, 0.9, [st], size=15, bold=True, color=WHITE, fill=[TEAL, NAVY][k % 2],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    table(s, 0.55, 2.85, 7.4, [0.9, 3.6, 2.9], [["Config", "Training data", "Selection data"],
          ["A", "English, human-labelled", "English val"], ["B", "+ translated Hindi/Bengali", "+ translated val"],
          ["C", "+ synthetic Hindi/Bengali", "+ synthetic val"]], size=16, row_h=0.6)
    text(s, 8.2, 2.85, 4.55, 3.95, [[("Models", {"bold": True})], "IndicBERT v2 (main), MuRIL, TF-IDF + LogReg baseline",
                                    [("Input", {"bold": True})], "(English claim, comment) pair, max 128 tokens",
                                    [("Training", {"bold": True})], "Class-weighted cross-entropy, lr 2e-5, batch 32, up to 5 epochs, "
                                    "10% warmup, mixed precision, seed 42"], size=16, fill=PANEL, pad=0.18, space=5)
    text(s, 0.55, 5.25, 7.4, 1.5, ["Class weights: Against is rare, so its errors count more.",
                                   "The claim in every input makes the model judge stance toward it, not just the topic."],
         size=17, color=MUTED, pad=0, space=6)

    # 6 implementation
    s = new_slide(6, "3 | Implementation", "Running code: one script per step, all on a laptop GPU",
                  "Demo live: python src/demo.py --examples, and python src/train_transformer.py --smoke (about 75 s).")
    table(s, 0.55, 1.6, 7.6, [2.6, 3.0, 2.0], [["Step", "Script", "Ran on"],
          ["Data splits", "make_splits.py", "laptop"], ["Synthetic data + judge", "generate_synthetic.py", "Gemini API"],
          ["Translation", "translate.py (NLLB-200)", "GPU"], ["Training A/B/C", "run_all.py, train_transformer.py", "GPU"],
          ["Evaluation", "evaluate.py", "laptop"], ["Human review", "apply_review.py", "team"],
          ["Live demo", "demo.py", "laptop"]], size=15, row_h=0.6)
    rinfo = [r["run"] for r in runs.values() if r.get("run", {}).get("device") == "cuda"]
    mins = [r["training_minutes"] for r in rinfo if "training_minutes" in r]
    gpu = next((r["gpu"] for r in rinfo if r.get("gpu")), "GPU")
    facts_panel = [[("Facts", {"bold": True, "color": NAVY})], f"{len(runs)} training runs on {gpu}"]
    if mins:
        facts_panel.append(f"Transformer runs took {min(mins):.1f} to {max(mins):.1f} minutes each")
    facts_panel += ["Every script ends with a built-in self-check", "Each run writes its predictions; every number in this deck "
                    "and the report is generated from them, never typed", "Demo shows the label and class probabilities for any comment"]
    text(s, 8.4, 1.6, 4.35, 5.2, facts_panel, size=16, fill=PANEL, pad=0.2, space=8)

    # 7 results table
    s = new_slide(7, "4 | Results", "Test macro-F1 for every model and configuration",
                  "Majority floor = always predicting the most common class. The synthetic column favours C (same generator).")
    order = [f"{m}_{c}" for m in ("tfidf", "muril", "indicbert") for c in ("A_english", "B_translated", "C_full")]
    rows = [["Model", "Cfg"] + [lab + (star if sk == "synthetic_test" else "") for sk, lab in SETS]]
    for name in order:
        if name in runs:
            rows.append([MODEL[runs[name]["model"]], runs[name]["config"][0]] +
                        [f"{get(name, sk):.3f}" if get(name, sk) is not None else "-" for sk, _ in SETS])
    floor = {sk: next(r["eval"][sk]["floor_macro_f1"] for r in runs.values() if sk in r["eval"]) for sk, _ in SETS
             if any(sk in r["eval"] for r in runs.values())}
    rows.append(["Majority floor", "-"] + [f"{floor[sk]:.3f}" if sk in floor else "-" for sk, _ in SETS])
    table(s, 0.55, 1.55, 12.2, [2.6, 0.8] + [1.75] * 5, rows, size=14, row_h=0.42)
    b = R["best_run"]
    text(s, 0.55, 6.15, 12.2, 0.7, [f"Best run (mean over translated Hindi, translated Bengali and pilot): "
                                    f"{MODEL[runs[b]['model']]} {runs[b]['config'][0]}." + ("  " + unverified if unverified else "")],
         size=14, color=MUTED, pad=0)

    # 8 findings
    s = new_slide(8, "4 | Results", "Three findings",
                  "Bars show 95% bootstrap confidence intervals; overlapping intervals mean no clear difference.")
    image(s, OUT + "fig_r2_macro_f1.png", 0.55, 1.6, w=7.3)
    tr_sets = ("translated_hi_test", "translated_bn_test")
    mean2 = lambda n: sum(get(n, x) for x in tr_sets) / 2  # noqa: E731
    finds = []
    if all(n in runs for n in ("tfidf_A_english", "indicbert_A_english")):
        finds.append(("Multilingual pre-training transfers", f"Trained on English only: TF-IDF {mean2('tfidf_A_english'):.2f} vs "
                      f"IndicBERT v2 {mean2('indicbert_A_english'):.2f} on translated Hindi/Bengali"))
    if all(n in runs for n in ("indicbert_A_english", "indicbert_B_translated")):
        finds.append(("Translated data is the big gain", "IndicBERT v2 B vs A: Hindi "
                      f"{get('indicbert_B_translated', tr_sets[0]) - get('indicbert_A_english', tr_sets[0]):+.3f}, Bengali "
                      f"{get('indicbert_B_translated', tr_sets[1]) - get('indicbert_A_english', tr_sets[1]):+.3f}"))
    if all(n in runs for n in ("indicbert_B_translated", "indicbert_C_full")):
        same = all(overlap(runs["indicbert_B_translated"]["eval"][x]["ci95"], runs["indicbert_C_full"]["eval"][x]["ci95"]) for x in tr_sets)
        finds.append(("Synthetic data: no clear gain on fair tests", "C vs B: Hindi "
                      f"{get('indicbert_C_full', tr_sets[0]) - get('indicbert_B_translated', tr_sets[0]):+.3f}, Bengali "
                      f"{get('indicbert_C_full', tr_sets[1]) - get('indicbert_B_translated', tr_sets[1]):+.3f}"
                      + ("; intervals overlap" if same else "")))
    for k, (head, body) in enumerate(finds):
        text(s, 8.1, 1.6 + k * 1.75, 4.65, 1.6, [[(head, {"bold": True, "color": NAVY})], body], size=16, fill=PANEL, pad=0.15, space=5)

    # 9 error analysis
    s = new_slide(9, "5 | Error analysis", f"Where the best run fails: {MODEL[runs[b]['model']]} {runs[b]['config'][0]}",
                  "Examples and notes from outputs/error_examples.csv (annotated by Aaditya).")
    if os.path.exists(OUT + "fig_r2_confusion_best.png"):
        image(s, OUT + "fig_r2_confusion_best.png", 0.55, 1.5, w=12.2)
    ev = runs[b]["eval"]
    ph = ev.get("synthetic_test", ev.get("pilot", {})).get("accuracy_by_phenomenon", {})
    worst = sorted(ph.items(), key=lambda kv: kv[1]["accuracy"])[:3]
    lang = {x: ev[x].get("macro_f1_by_language", {}) for x in ("synthetic_test", "pilot") if x in ev}
    left = [[("Weakest phenomena (accuracy)", {"bold": True, "color": NAVY})], ", ".join(f"{k.replace('_', ' ')} {v['accuracy']:.2f}" for k, v in worst)]
    if lang:
        left += [[("Hindi vs Bengali (macro-F1)", {"bold": True, "color": NAVY})],
                 "; ".join(f"{SET_LABEL.get(x, x)}: hi {d['hi']:.2f}, bn {d['bn']:.2f}" for x, d in lang.items() if "hi" in d and "bn" in d)]
    text(s, 0.55, 4.3, 5.4, 2.55, left, size=16, fill=PANEL, pad=0.15, space=5)
    ex_path = OUT + "error_examples.csv"
    if os.path.exists(ex_path):
        ex = pd.read_csv(ex_path, dtype=str, keep_default_na=False)
        ex = ex[ex.get("team_notes", pd.Series("", index=ex.index)).str.strip() != ""]
        ex = ex.assign(L=ex.comment.str.len()).sort_values("L").head(2)
        paras = []
        for _, r in ex.iterrows():
            paras += [[(f"{r['comment']}", {"italic": True})],
                      [(f"true {r['gold']}, predicted {r['pred']}: ", {"bold": True}), (r["team_notes"].split(". ")[0].rstrip(".") + ".", {})]]
        if paras:
            text(s, 6.15, 4.3, 6.6, 2.55, paras, size=14, fill=PANEL, pad=0.15, space=4)

    # 10 contributions, challenges, next steps
    s = new_slide(10, "6 | Team and next steps", "Who did what, what was hard, what comes next",
                  "Each member commits their own work from their own account and explains their part.")
    table(s, 0.55, 1.5, 12.2, [2.3, 9.9], [["Member", "Responsibility"],
          ["Sumukh Gupta", "Models and training: splits, configurations, IndicBERT v2 / MuRIL / TF-IDF runs, demo pipeline"],
          ["Neya Tripathi", "Dataset generation and translation; Hindi review of the synthetic test labels" + (" and translations" if tr else "")],
          ["Aaditya Goel", "Evaluation and error analysis: metrics, confidence intervals, annotated misclassifications"],
          ["Kushika Mishra", "Progress report, deck and live demo testing"],
          ["Eklavya Sharma", "Bengali review of the synthetic test labels" + (" and translations" if tr else "") + "; Hindi-vs-Bengali analysis"]], size=15, row_h=0.42)
    text(s, 0.55, 4.25, 5.95, 2.6, [[("Challenges", {"bold": True, "color": CORAL})],
                                    "No real labelled Hindi/Bengali comments: every Indic test set is a proxy",
                                    "Synthetic test shares a generator with C's training data",
                                    "Sarcasm and indirect stance are ambiguous even for the judge model"], size=16, fill=PANEL, pad=0.15, space=5)
    nxt = [[("Next steps", {"bold": True, "color": TEAL})]]
    if not F["verified"]:
        nxt.append("Finish the blind review of the 600 synthetic test comments")
    if not tr:
        nxt.append("Hand-check that the stance survives machine translation")
    nxt += ["Label a small set of real public comments as a true test set",
            f"Target the weakest cases ({F['weak']}) with more varied data", "Several seeds per run; zero-shot LLM baseline"]
    text(s, 6.8, 4.25, 5.95, 2.6, nxt, size=16, fill=PANEL, pad=0.15, space=5)

    assert len(prs.slides) == TOTAL
    prs.save(OUT + "Round2_Deck.pptx")
    print(f"saved {OUT}Round2_Deck.pptx with {len(prs.slides)} slides; fit warnings: {len(WARN)}")
    for w in WARN:
        print("WARNING", w)


SET_LABEL = {"synthetic_test": "synthetic", "pilot": "pilot"}


if os.path.exists(OUT + "round2_results.json"):
    round2_deck()
else:
    print("outputs/round2_results.json not found: run src/evaluate.py to build the Round 2 deck")
