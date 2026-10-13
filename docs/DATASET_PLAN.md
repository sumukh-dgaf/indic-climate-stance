# Dataset plan, provenance and status

Companion to [`data/DATASET_INVENTORY.md`](../data/DATASET_INVENTORY.md) (one row per source, gap table).
This file records provenance, how files were normalised, what is deliberately **not** done, and the Milestone 2 plan.

## 1. Honest status

**Usable dataset found: PARTIALLY.**

* English, target-correct, labelled text exists: GWSD + SemEval-2016 climate (2,408 rows, supplied by the team and re-checked by the validation gate).
* **No Hindi/Bengali climate-stance comment data was obtainable.** The official AISoMe 2026 data was emailed to an address we cannot reach.
* The only Hindi/Bengali comment-style data in this repository is the **team-authored pilot corpus** (section 4). It is synthetic.
  Any number computed on it is labelled "pilot corpus, development only - not comparable to official-set results".

## 2. Provenance and files

| File | Origin | Licence / terms | Mapping |
|---|---|---|---|
| `data/english_stance_merged.csv` | team-supplied merge of GWStance/GWSD (Luo et al. 2020) and SemEval-2016 climate | not re-checked by us; **not committed to git** | majority vote of 8 annotators (GWSD); published gold (SemEval) |
| `data/english_stance_highconf.csv` | subset: GWSD agreement >= 0.75 plus all SemEval | as above; not committed | as above |
| `data/gwstance_ambiguous_ties.csv` | the 202 GWSD rows with no annotator majority (written by `src/data.py`) | as above; not committed | no label |
| `data/external/semeval2016_climate_en.csv` | `cardiffnlp/tweet_eval` `stance_climate` (SemEval-2016 Task 6) | check the dataset card; research use | favor/against/none |
| `data/external/dhoroni_bn_news.csv` | `ciol-research/Dhoroni` | MIT (dataset card) | Positive->Favour, Negative->Against, Neutral->None, Problematic dropped (107) |
| `data/external/multiclimate_en_video.csv` | github.com/werywjw/MultiClimate | see repo LICENSE (not read in detail) | support->Favour, oppose->Against (**low confidence**), neutral->None |
| `data/external/swami_hinglish_demonetisation.csv` | github.com/sahilswami96/StanceDetection_CodeMixed | GPL-3.0 (LICENSE header) | FAVOR/AGAINST/NONE -> Favour/Against/None **about demonetisation** |
| `data/pilot_indic_stance.csv` | written by the team | ours | native labels |
| `data/raw/` | untouched downloads | upstream | - |

Normalised external files share one schema: `id, comment, stance, language, target, domain, source, label_original`.
Sources are never concatenated into one file without a `source` column. `data/external/` and `data/raw/` are git-ignored
(third-party licences; `python src/fetch_external.py` rebuilds them).

### Why two label mappings are risky

* **MultiClimate "oppose".** The paper defines Oppose as "negative sentiments or criticism towards climate change or its
  related aspects". A transcript sentence that laments climate harm is therefore "oppose" even though it *agrees* the issue
  is serious. Mapping oppose->Against inverts those sentences. We map it for completeness but flag it low-confidence and
  do not train on it.
* **Swami et al. 2018.** Same three label names, different target (demonetisation). It is a textbook **cross-target**
  stance problem: "Favour" there means *favours demonetisation*. Mixing it into the climate pool would teach the
  model that a Romanised-Hindi comment praising a government policy is climate-Favour. It is kept in its own file.

### Domain mismatches (stated, not hidden)

* Dhoroni is Bengali **news articles** (median 331 words), not comments.
* GWSD is **news sentences**: 1,845 of the 2,408 labelled rows (77%); only 563 rows are tweets.
* MultiClimate is **video transcript sentences**.

## 3. Quality gate

`src/data.py::validate` runs before every training/evaluation use of a labelled file and raises if it finds: empty or NaN
labels, labels outside {Favour, Against, None}, duplicate comments, empty comments, or empty `source`. It prints the
per-source class distribution. The loader also keeps the literal label `"None"` (`keep_default_na=False`; a plain
`pd.read_csv` turns it into NaN and deletes a class), splits the legacy `agreement` column into `label_source`
(`majority_vote_8` | `published_gold`) and a numeric `agreement`, and raises on any unmapped label.

### The 202 tie rows

202 of the 2,047 GWSD sentences (9.9%) have no annotator majority (147 split 4-4, 55 split 3-3-2). They are dropped
from training and evaluation and written to `data/gwstance_ambiguous_ties.csv`. We report them as evidence of genuine task
difficulty: even eight crowd annotators cannot settle a tenth of the sentences. The supplied file may have had further
tie rows removed before it reached us (earlier note: 215); we cannot verify that, so we quote 9.9% as a share of the rows we
can see and treat it as indicative, not as a formal ceiling on macro-F1.

## 4. The pilot corpus (team-authored, development only)

* `data/pilot_indic_stance.csv`: 150 comments, 25 per class per language (Hindi, Bengali).
* **Written by the team because the official dataset was not in hand. It is synthetic: not scraped, not real user data,
  and not the official dataset.** The drafting was assisted by an LLM and should be read and corrected by native speakers.
* Phenomena covered on purpose (what the hidden test set is expected to probe): plain 28, roman 22, codemix 22,
  offtopic 16, question 16, implicit 14, negation 12, emoji 12, sarcasm 8.
* Includes the realistic Against rhetoric (always hot, Western conspiracy, scientists want funding, cold wave so no warming)
  and None traps (nice video, link?, timestamps, subscribe).
* Use: check that the pipeline handles Devanagari, Bengali, Romanised and code-mixed text; develop error analysis.
  It must not be used to claim performance on the official data.

## 5. Milestone 2 plan: bridging English training data to Indic test data

Two candidate strategies; both are **planned, not implemented** in Milestone 1.

| | Zero-shot cross-lingual transfer | Translate-train |
|---|---|---|
| What | Fine-tune a multilingual encoder (MuRIL / XLM-R) on English, evaluate directly on Hindi/Bengali | Machine-translate the English training set into Hindi and Bengali (IndicTrans2 or NLLB) and train on the translations (optionally together with the English) |
| For | No translation cost; no translation noise; relies on shared multilingual representations | Supplies in-language supervision, including Devanagari/Bengali script and vocabulary |
| Against | No in-language supervision; degraded on Romanised and code-mixed text the encoder saw little of | Translation noise (MT can flip negation and lose sarcasm); **news-register sentences become formal Hindi, unlike YouTube comments**; stance cues partly lost |
| Evidence | IndicXNLI: English + Indic combined training worked best (supports using both) | Squad4 (a competing AISoMe 2026 team; see literature review) used translate-train and reported a large gap between internal validation and official test scores (a warning that internal CV is optimistic) |

Planned ablation for Milestone 2: English-only (zero-shot) vs translated-only vs English + translated, each evaluated
per language and per phenomenon, with the pilot used only for development. Optional additions: transliteration of
Romanised text to native script before encoding; domain-adaptive pretraining on the Swami corpus and Dhoroni text
(unlabelled use, so the target mismatch does not matter).

Hindi Climate-FEVER may be used for unlabelled Hindi climate-domain continued pretraining; it supplies no stance labels.
