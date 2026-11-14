# Overnight log (Round 2)

## STATUS (top of file, kept current)

**Last update:** 2026-11-14 17:30 IST
**Done:** Stage 1 code (pushed); NLLB translation of all English splits; IndicBERT v2 A_english.
**Running:** Gemini synthetic generation (resumable); A/B training schedule (`src/run_all.py`).
**Not started:** C_full runs (blocked until `data/synthetic/indic_synthetic_v1.csv` is complete), Stage 2 final outputs.
**What you need to do next:** nothing yet; this section is updated at every step.

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
* Gemini returns frequent HTTP 503 (server overload); the script backs off 30-300 s and retries. ~6 batches/min.

## Problems and what was done

* HTTP 503 from Gemini: handled by back-off; no data lost.
* Heredoc quoting failed when writing the task files: wrote them directly instead.
