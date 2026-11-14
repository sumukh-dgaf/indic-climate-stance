# Overnight log (Round 2)

## STATUS (top of file, kept current)

**Last update:** 2026-11-14 18:50 IST. **Everything in Stage 1 and Stage 2 is done and pushed. Nothing is running.**

**Done:** synthetic dataset (2,901 rows, committed); NLLB translation; all 8 scheduled runs (IndicBERT v2 A/B/C,
TF-IDF A/B/C, MuRIL A/C) on the RTX 5050; evaluation (`outputs/round2_results.md/.json`, figures, 15 error
examples); two-page report `docs/Round2_Progress_Report.docx` (+ .pdf; exactly 2 A4 pages by Word's count);
9-slide deck `outputs/Round2_Deck.pptx`; README results table + provenance + reproduce steps; CONTRIBUTIONS.md and
`docs/tasks/<name>.md`; demo tested with the C model. All 11 self-checks pass. The Gemini daily quota was never hit.

**What you need to do next (in order):**
1. Open `docs/Round2_Progress_Report.pdf` and `outputs/Round2_Deck.pptx` and read them once.
2. GitHub > Settings > Collaborators: add Neya, Aditya and Kushika so they can push from their own accounts.
3. Share privately (not via git): `data/review/translation_review_hi.csv` and `_bn.csv` (GWStance text) with Neya and
   the Bengali reader; `models/indicbert_C_full/` (~1 GB) with Kushika for the demo.
4. Send each teammate their file in `docs/tasks/`. The review must be done before the synthetic test numbers can be
   called human-verified; after it, run `python src/apply_review.py`, `python src/evaluate.py`,
   `python src/make_report.py`, `python src/make_deck.py` (then `git checkout -- outputs/Milestone1_Sumukh_S24CSEU0570.pptx`,
   which make_deck.py rebuilds unchanged).
5. Rehearse the demo: `python src/demo.py --examples` (8 s on CPU), then 2 or 3 comments of your own.

## Final results (from `outputs/round2_results.json`; macro-F1, 95% bootstrap CI in the results file)

| Run | English (349) | Transl. hi (349) | Transl. bn (349) | Synthetic* (600) | Pilot (150) |
|---|---|---|---|---|---|
| TF-IDF A | 0.602 | 0.335 | 0.341 | 0.285 | 0.341 |
| TF-IDF B | 0.592 | 0.580 | 0.600 | 0.435 | 0.537 |
| TF-IDF C | 0.631 | 0.600 | 0.600 | 0.832 | 0.781 |
| MuRIL A | 0.646 | 0.586 | 0.552 | 0.592 | 0.647 |
| MuRIL C | 0.704 | 0.655 | 0.641 | 0.864 | 0.824 |
| IndicBERT v2 A | 0.717 | 0.581 | 0.566 | 0.414 | 0.643 |
| IndicBERT v2 B | 0.732 | 0.700 | 0.678 | 0.577 | 0.715 |
| **IndicBERT v2 C** | 0.705 | **0.703** | **0.693** | 0.875 | **0.858** |
| Majority floor | 0.238 | 0.238 | 0.238 | 0.167 | 0.167 |

\* synthetic test labels not yet human-verified. Best run by the pre-set rule (mean of translated hi, translated bn,
pilot): IndicBERT v2 C. Training time on the RTX 5050: IndicBERT C 4.1 min, B 3.0, A 1.0; MuRIL C 3.8, A 1.0.

**Caveats a grader can ask about:**
* No score is on real Hindi/Bengali comments (translated, synthetic, or the synthetic pilot).
* Synthetic data adds **no clear gain on the translated sets** (IndicBERT C vs B: +0.002 hi, +0.015 bn; CIs overlap).
  C's big gains are on the synthetic test (same generator: optimistic) and the pilot (also synthetic comment-style).
* The judge filter keeps only comments a second LLM agrees with, so the synthetic test is easier than real data
  (only 36.9% of sarcasm survived it). C's 0.82 sarcasm accuracy on it is not evidence about real sarcasm.
* IndicBERT A predicts Favour for 461 of 600 synthetic comments (reads any climate mention as agreement).
* English test has only 39 Against rows; single seed; translated labels are carried over from English, unreviewed.
* IndicBERT C is overconfident: probabilities of 0.97-1.00 on most demo examples; it gets the sarcastic Hindi
  example wrong (Favour 0.80).

---

---

## Decisions (made unattended, with reasons)

* Training runs locally on the RTX 5050 instead of Colab (your instruction). The Colab notebook stays in the repo.
* Generator `gemini-3.5-flash-lite`, judge `gemini-3.1-flash-lite`: both free-tier per Google's pricing page; a
  different judge model so it does not grade its own output.
* C_full is gated in code, not by convention: `data.py` loads synthetic data only when `generation_stats.json` says
  `"complete": true`; an interrupted generation writes `indic_synthetic_v1.partial.csv`, which nothing trains on.
* A/B transformer checkpoints are kept until their synthetic-test predictions exist (`run_all.py` backfills them once
  the synthetic data is complete), so A and B are scored on the same five test sets as C.
* Error examples come only from our own data (synthetic test + pilot): translated/English rows contain GWStance text,
  which has no licence. `data/review/translation_review_*.csv` is git-ignored for the same reason.
* Best run for confusion matrices / error analysis = highest mean macro-F1 on translated Hindi, translated Bengali and
  pilot (synthetic test excluded because C trains on the same generator). Rule fixed before seeing C results.

## Timeline

* **17:05** Gemini key found in `.env`. Generation started; the first calls succeeded.
* **17:16** Stopped the first `run_all.py` before any training run (it would have deleted A/B checkpoints before the
  synthetic test existed). Added the completeness gate (commit 516f4ec) and restarted generation on the new code;
  it resumed from the 22 batches already on disk (no calls lost).
* **17:20** Translation finished with `facebook/nllb-200-distilled-600M` (no HF_TOKEN, so IndicTrans2 was not tried).
  Rows: hi/bn train 1,750, val 309, test 349 each. Time: hi 127 s, bn 194 s on the RTX 5050. Share of rows whose
  script is purely native: hi 0.949 / 0.955 / 0.874 (train/val/test), bn 0.952 / 0.961 / 0.885; the rest keep
  Latin-script tokens (hashtags, names, acronyms) and count as "mixed".
* **17:23** IndicBERT v2 A_english: best val macro-F1 0.662 (epoch 5), 1.3 min. Quick test macro-F1: English 0.717,
  translated hi 0.581, translated bn 0.566, pilot 0.643 (authoritative numbers come from `src/evaluate.py`).
* Gemini returns frequent HTTP 503 (server overload); the script backs off 30-300 s and retries. ~3 batches/min.
* **17:27** A/B schedule finished (quick test macro-F1 from each run; `evaluate.py` gives the authoritative ones, below):
  IndicBERT v2 B_translated: val 0.674 (epoch 4), 3.3 min. TF-IDF A: val 0.630. TF-IDF B: val 0.606.
  MuRIL A: val 0.630, 1.2 min. C runs correctly skipped ("waiting for complete synthetic data").
* **17:40** Provisional `evaluate.py` on the A/B runs (synthetic test not yet available), macro-F1:

  | Run | English | Transl. hi | Transl. bn | Pilot |
  |---|---|---|---|---|
  | TF-IDF A | 0.602 | 0.335 | 0.341 | 0.341 |
  | TF-IDF B | 0.592 | 0.580 | 0.600 | 0.537 |
  | MuRIL A | 0.646 | 0.586 | 0.552 | 0.647 |
  | IndicBERT v2 A | 0.717 | 0.581 | 0.566 | 0.643 |
  | IndicBERT v2 B | 0.732 | 0.700 | 0.678 | 0.715 |
  | Majority floor | 0.238 | 0.238 | 0.238 | 0.167 |

  IndicBERT v2 B on pilot: lowest phenomenon accuracy sarcasm 0.38 (n=8), roman 0.50 (n=22). Report draft = exactly
  2 pages (Word page count). Deck dry run = 9 slides.
* Decision: the demo now falls back to IndicBERT v2 B (then TF-IDF) if the C model is missing; the B checkpoint is kept.
  With B, the demo's fixed examples: 10 of 11 match the stance they were written with.

* **17:47** All 200 planned generation batches done: **4,000 comments generated**. API-free filters: 3 removed for
  length/format, **660 for script**, 2 exact duplicates, 18 near-duplicates -> 3,317 go to the judge
  (hi Favour 560 / Against 558 / None 545; bn Favour 563 / Against 558 / None 533).
* **17:50 Finding + fix (code-mixed prompt).** 593 of the 660 script failures were `codemix` comments: 593 of 600
  came back entirely in Latin letters (Romanised Hinglish/Banglish) instead of native script mixed with English. The
  generator was stopped during the judge pass (50 comments judged; nothing lost), the code-mixed instruction was made
  explicit (native script required, one style example, "do not reuse that sentence"), and each new batch now stores its
  exact prompt in `generate.jsonl`. Test batch `hi|Favour|codemix|5` with the new prompt: **20 of 20 passed** the
  script check. The rejected Romanised rows are not relabelled as `roman`; they are simply filtered out (documented).
  The automatic top-up rounds regenerate the missing code-mixed rows with the new prompt.
* **17:52** Generation restarted: judge pass on the 3,317 comments, then top-ups, then (chained) `run_all.py`.
* **18:12** Judge pass finished (`gemini-3.1-flash-lite`, sees only claim + comment). Agreement with the intended label
  on 3,337 judged comments: **88.3%** overall; by label Against 96.2%, Favour 86.2%, None 82.4%; by language hi 89.4%,
  bn 87.2%; by phenomenon sarcasm **58.1%** (lowest), question 81.8%, plain 89.8%, implicit 90.3%, neutral_fact 90.3%,
  emoji 91.3%, roman 92.5%, offtopic 93.7%, negation 99.1%, codemix 100% (only the 20 new-prompt rows so far).
  Disagreements (intended -> judge): Favour -> Against 107 (mostly sarcastic pro-climate comments whose literal words
  sound like denial), None -> Favour 123, None -> Against 67, Favour -> None 51. 390 rows removed.
  Top-up round 1: 538 rows short in 11 cells (6 code-mixed cells, Favour-sarcasm hi 36 / bn 44, None-plain hi 6 / bn 17,
  None-emoji bn 12) -> 46 batches.
* **18:28 Synthetic dataset COMPLETE** (`data/synthetic/indic_synthetic_v1.csv`, `generation_stats.json`
  `"complete": true`). Final pipeline counts: **5,180 generated** (200 planned + 46 round-1 + 13 round-2 top-up
  batches, plus the 1 test batch) -> removed 3 length/format, **878 script**, 2 exact duplicates, 19 near-duplicates,
  **721 judge disagreements** (4,278 judged, **agreement 83.1%**) -> **3,557 passed all filters** -> **2,901 kept**
  after the quota cap -> 14 train rows dropped as near-duplicates of val/test rows.
  Split: **test 600** (exactly 100 per label per language), **val 300** (50 each), **train 2,001**.
  Shortfall vs the 3,000 target: **85 rows** in 4 cells: Favour-sarcasm hi 27 / bn 25; None-codemix hi 9 / bn 24.
  Final judge agreement by label: Against 96.6%, Favour 75.3%, None 79.1%; by phenomenon: **sarcasm 36.9%**, question
  81.8%, codemix 87.1%, plain 89.2%, emoji 90.1%, implicit 90.3%, neutral_fact 90.3%, roman 92.5%, offtopic 93.7%,
  negation 99.1%. The overall rate fell from 88.3% (first pass) to 83.1% because the top-ups deliberately
  over-generated the hardest cell (Favour sarcasm); per-phenomenon rates are the fairer reading.
  Phenomena kept: plain 748, roman 449, codemix 417, emoji 257, implicit 240, negation 239, sarcasm 188, offtopic 137,
  neutral_fact 119, question 107. Every row: source "synthetic (LLM-generated)", generator `gemini-3.5-flash-lite`,
  human_verified "no". Review file `data/review/synthetic_test_review.csv` (600 rows, blind) written.
  Daily quota was **not** exhausted; no wait for the Pacific-midnight reset was needed.
* **18:28** C runs started (chained `run_all.py`): IndicBERT v2 C, TF-IDF C, MuRIL C, then the synthetic-test
  backfill for the A/B runs.
* Fixed: the stats file stored the judge confusion matrix transposed ({judge: {intended}}) under an "intended vs judge"
  key; now {intended: {judge}}. Also removed a pandas FutureWarning. Data unaffected.

* **18:37** All C runs finished (IndicBERT C 4.4 min wall, TF-IDF C 0.1, MuRIL C 4.0) and the A/B runs were
  back-filled with synthetic-test predictions (`predict.py --eval-all` on the kept checkpoints; TF-IDF refit). 40
  prediction files = 8 runs x 5 test sets. The A/MuRIL checkpoints were then deleted; `models/indicbert_C_full` (demo)
  and `models/indicbert_B_translated` (demo fallback) are kept.
* **18:38** `generate_synthetic.py --no-api` rebuild with the fixed confusion-matrix orientation: the dataset CSV and
  the review file came out **byte-identical** (sha1), so C trained on exactly the committed data.
* **18:40** Final `evaluate.py`. Best run: IndicBERT v2 C. Report: 11 pt overflowed to 3 pages, 10.5 pt = 2 pages.
* Two hand-written claims were caught and replaced by computed ones: the report said the judge disagreed "most often
  for sarcasm and implicit stance" (implicit was actually 90.3%); now generated: sarcasm 36.9%, question 81.8%.
  The weak-phenomena lists in the report and deck are now computed from the best run (roman, sarcasm, negation).
* Added limitations: judge filter makes the synthetic test easier; the pilot is synthetic too; single seed.
* **18:45** Demo with the C model: 10 of 11 fixed examples match (misses the sarcastic Hindi one), 8 s on CPU.

## Problems and what was done

* HTTP 503 from Gemini: handled by back-off; no data lost.
* Heredoc quoting failed when writing the task files: wrote them directly instead.
