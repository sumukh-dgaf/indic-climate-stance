# Detecting Stance in Media on Global Warming for Indic Languages

CSET346 Natural Language Processing, Bennett University. **Milestone 1** (problem analysis, literature review, dataset
exploration, methodology, baseline).
Author: Sumukh Gupta (S24CSEU0570). Instructor: Dr. Supriya Chanda.

## Problem

Classify a Hindi or Bengali YouTube / social-media comment by its stance toward the claim

> **"Climate change and global warming is a serious concern."**

| Label | Meaning |
|---|---|
| **Favour** | the comment agrees the claim is a serious concern |
| **Against** | the comment disagrees (denies, minimises, mocks the concern) |
| **None** | neutral, unrelated, or no clear stance |

Primary metric: **macro-F1**. Supporting: accuracy, per-class precision/recall, confusion matrix.

The task is derived from the **AISoMe 2026 shared task at FIRE 2026** (same claim, labels and languages). Its defining
property: **public training data is English; the test data is Hindi and Bengali.** That makes this a **cross-lingual
transfer** problem, not plain text classification. See [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md).

## Data status (read this first)

**Usable dataset found: PARTIALLY.**

* **Real, English, target-correct:** 2,408 labelled sentences from GWStance/GWSD (news) and SemEval-2016 Task 6 climate
  (tweets). Supplied as `data/english_stance_merged.csv`; 202 annotator-tie rows (215 of 2,300 = 9.3% in the raw GWSD
  file; 13 removed by deduplication) are removed by the loader.
* **No Hindi/Bengali stance comments on this claim were obtainable.** The official AISoMe data was not in hand.
* **`data/pilot_indic_stance.csv` is a team-authored pilot corpus: 150 synthetic Hindi/Bengali comments written
  because the official dataset was not available. It is not scraped, not real user data and not the official dataset.**
  Any score on it is "pilot corpus, development only - not comparable to official-set results".
* External sources examined (Dhoroni, MultiClimate, ClimaConvo, Swami et al.) are documented in
  [`data/DATASET_INVENTORY.md`](data/DATASET_INVENTORY.md), including why each is not in the training pool. The
  code-mixed Hindi corpus is about **demonetisation**, not climate, and is kept separate.
* Provenance, label mappings and the Milestone 2 translate-train plan: [`docs/DATASET_PLAN.md`](docs/DATASET_PLAN.md).

Third-party data is git-ignored. The GWStance repository has **no LICENSE file**, so `data/english_stance_*.csv` and
`data/gwstance_ambiguous_ties.csv` are not committed; `python src/build_english_merged.py` rebuilds them from source and its
`--check` mode verified a field-by-field match with the team's files. Exact rules: [`data/README.md`](data/README.md).

## Results so far (all measured by code in this repo)

English merged, 2,408 rows, stratified 5-fold CV, macro-F1 (`outputs/baseline_results.md`):

| Model | macro-F1 |
|---|---|
| Majority-class floor | 0.200 |
| TF-IDF word 1-2 + LogReg | 0.588 |
| TF-IDF char_wb 2-5 + LogReg | 0.605 |
| **TF-IDF word + char + LogReg (class-weighted)** | **0.613 +- 0.025** |
| same, no class weights | 0.558 |

Pilot corpus (**development only, synthetic**): 5-fold CV 0.57-0.58; zero-shot English -> pilot 0.341
(floor 0.167), predicting "None" for 131 of 150 comments: a TF-IDF model trained on English has almost nothing to
transfer, which is the cross-lingual gap a multilingual encoder must close.

The transformer fine-tuning script is written but **has not been trained** in Milestone 1.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10. The transformer script additionally needs `torch`, `transformers`, `sentencepiece` (in requirements).

## Reproduce

```bash
python src/preprocess.py            # self-check: NFC, repeat collapse, script ID
python src/data.py                  # self-check: loader, "None" label gotcha, validation gate
python src/build_english_merged.py  # rebuilds data/english_stance_*.csv from GWSD + SemEval (needs internet)
python src/fetch_external.py        # downloads + normalises external sources into data/external/ (needs internet)
python src/eda.py                   # outputs/eda_report.md, eda_stats.json, fig_*.png
python src/baseline.py              # outputs/baseline_results.md/.json, confusion matrices
python src/make_deck.py             # outputs/Milestone1_Sumukh_S24CSEU0570.pptx (reads the JSON above)
```

Milestone 2 (not run in Milestone 1):

```bash
python src/train_transformer.py --smoke                 # CPU smoke test
python src/train_transformer.py --model muril --epochs 4   # Colab GPU; also: indic-bert, xlm-r
python src/predict.py --model-dir outputs/models/muril --input test.csv --output outputs/submission.csv
```

## Repository structure

```
README.md
requirements.txt
data/
  pilot_indic_stance.csv        team-authored synthetic pilot (development only)
  DATASET_INVENTORY.md          every source examined, status, gap table
  README.md                     how the git-ignored English files are rebuilt, and why they are not committed
  english_stance_*.csv          supplied English files (git-ignored)
  external/, raw/               fetched third-party data (git-ignored)
docs/
  LITERATURE_REVIEW.md          fundamentals -> climate -> cross-lingual -> Indic -> LLM -> SOTA -> gap -> method
  DATASET_PLAN.md               provenance, mappings, quality gate, translate-train vs zero-shot
  VIVA_PREP.md                  answers to the instructor's questions + follow-ups
src/
  preprocess.py  data.py  eda.py  baseline.py
  train_transformer.py  predict.py  fetch_external.py  build_english_merged.py  make_deck.py
outputs/
  eda_report.md  baseline_results.md  *.json  fig_*.png  Milestone1_*.pptx
```

Every module in `src/` carries `assert`-based self-checks (`python src/<module>.py`).

## Indic preprocessing findings

1. Apply Unicode NFC first: Devanagari nukta letters have precomposed and decomposed encodings.
2. Repetition collapse acts on code points, not glyphs: `नहीीीीी` becomes `नहीी` because the vowel sign is a code point.
3. Python `\w` / scikit-learn's default tokenizer split Hindi words at vowel signs; we use an Indic-aware pattern.
4. Script ID from Unicode ranges separates Devanagari/Bengali but cannot tell Romanised Hindi from English.
5. `pandas.read_csv` turns the label string `"None"` into NaN; always `keep_default_na=False`.

## Team and contributors

| Name | Enrolment number |
|---|---|
| Sumukh Gupta (repository owner) | S24CSEU0570 |
| Neya Tripathi | S24CSEU0609 |
| Aditya Goel | S24CSEU0614 |
| Kushika Mishra | S24CSEU0652 |

B.Tech CSE, Bennett University. Course instructor: Dr. Supriya Chanda.

## Next steps (Milestone 2)

1. Fine-tune MuRIL on `claim [SEP] comment` (XLM-R, IndicBERT as flagged comparisons).
2. Compare zero-shot transfer, translate-train (IndicTrans2/NLLB) and English + translated data.
3. Evaluate per phenomenon (negation, sarcasm, implicit, code-mixed, Romanised) and add a zero-shot LLM baseline.
4. Replace the pilot with the official data as soon as it is available; keep the pilot for error analysis only.
