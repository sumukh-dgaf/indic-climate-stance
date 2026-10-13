# Viva preparation: Milestone 1

Sumukh Gupta, S24CSEU0570. CSET346 NLP, Dr. Supriya Chanda.
Every number here is from code in this repo (`outputs/*.md`, `outputs/*.json`). If a number is not there, say "not yet measured".

**One-sentence pitch.** "Same vocabulary, different stance, and the training data is in the wrong language: this is a
cross-lingual transfer problem, so I measured the data honestly, built baselines with a floor, and designed a target-aware
multilingual transformer for Milestone 2."

---

## A. The instructor's questions

**1. What is stance detection?**
Deciding whether a text is in favour of, against, or neutral toward a *target*. The target is usually not in the text.
It is not sentiment: "Coal plants are a disaster" is negative in tone and *Favour* of the claim that climate change is serious.

**2. What is the claim?**
"Climate change and global warming is a serious concern." Every comment is judged relative to this sentence.

**3. What are Favour, Against and None?**
Favour: agrees it is a serious concern. Against: disagrees, minimises or denies it. None: neutral, unrelated or no clear
stance (so None mixes three things: neutral, off-topic and ambiguous).

**4. What makes the task hard?**
The label depends on a claim that never appears; the same words appear in Favour and Against; stance is often implicit;
and None is heterogeneous. On top: Hindi/Bengali script and spelling variation, Romanised text, code-mixing, sarcasm,
and **no in-language training data**.

**5. What linguistic challenges arise in Hindi/Bengali social media?**
* Three writing systems for one language: Devanagari/Bengali script, **Romanised** (`global warming sab jhooth hai`),
  and mixed with English (`Climate change का scam बंद करो`).
* **Spelling variation**: no standard Romanisation, so one word has many spellings. Measured: hapax rate (share of vocabulary
  types seen exactly once) is 54.3% on the English set and 68.4% on the pilot.
* Unicode traps: nukta has two encodings (NFC first); repetition collapse works on code points (`नहीीीीी` -> `नहीी`); Python's
  `\w` splits Hindi words at vowel signs, so we use an Indic-aware token pattern.
* Negation and sarcasm flip stance while the keywords stay (`इस ठंड में तो धरती और गरम हो रही है 😂` is Against).

**6. Why is this NOT simple text classification?** (say these four, in this order)
1. **The label is defined relative to an external claim that never appears in the comment.** "Cold wave in Delhi" is
   Against only because the claim is about warming.
2. **Favour and Against share vocabulary and differ only by negation, sarcasm or framing.** Measured: 58.9% of the
   vocabulary of Against sentences also occurs in Favour sentences. "Global warming is a hoax" and "Don't tell me global
   warming is a hoax" use the same words.
3. **Stance is often implicit and needs world knowledge.** "Our village well dried up in May" is Favour with no climate word
   in it. A keyword model cannot get this.
4. **None is not a topic, it is a junk drawer**: neutral, off-topic ("first comment") and ambiguous together.
   Plus an extra reason for this project: **train in English, test in Hindi/Bengali**, so even a perfect English classifier
   is evaluated out of language.

**7. What has already been done?**
SemEval-2016 Task 6 (the three-label formulation, with a climate target); GWSD (news sentences about global warming,
8 annotators each); ClimaConvo (climate tweets, but the repo gives IDs only); CASE@EACL 2024 (a climate-activism shared task;
the HAMiSoN-Ensemble entry combined RoBERTa + Llama 2 + multi-task learning and found that adding external data did NOT
help RoBERTa or Llama 2); MultiClimate (YouTube transcripts, multimodal); Dhoroni (Bangla news, BanglaBERT F1 0.308); Swami et al.
(Romanised Hindi stance, but on demonetisation, 58.7% accuracy); and AISoMe participants (translate-train with MuRIL/XLM-R,
official macro-F1 0.42-0.53).

**8. What did I discover from the data?**
* **Against is rare**: 26 of 564 SemEval climate tweets (4.6%), 14.3% of the 2,408-row English set.
* **215 of the 2,300 raw GWSD sentences (9.3%) have no annotator majority** (measured by our script on the raw file); 13
  were removed by deduplication, so 202 survive in our merged file, are dropped from training/evaluation and saved
  separately. It is a measured indication of annotator disagreement and task difficulty.
* **No existing resource matches target + language + domain** (gap table in `data/DATASET_INVENTORY.md`).
* **ClimaConvo is unusable as distributed**: 15,309 rows, tweet IDs only.
* **Label names lie across targets**: Swami's FAVOR means "favours demonetisation".
* Collection artefacts: 21 SemEval Favour tweets carry one conference hashtag (#CFCC15), so English CV is optimistic.
* A TF-IDF model trained on English scores macro-F1 0.341 on the Indic pilot and predicts "None" for 131 of 150 comments:
  almost no shared features (pilot = development only).

**9. Limitations of existing approaches?**
English-only or news-only target-correct data; the one Hindi stance corpus has the wrong target; text not distributable
(ClimaConvo); large validation-to-test drop in the shared task (0.65-0.68 -> 0.42-0.53); minority class recovered worst;
fine-tuned models are not guaranteed to beat zero-shot LLMs.

**10. My proposed idea?**
Target-aware (`claim [SEP] comment`) fine-tuning of MuRIL, with class-weighted loss and macro-F1 selection, comparing
zero-shot transfer from English to translate-train (IndicTrans2/NLLB) and English + translated data, evaluated per
phenomenon, behind a measured TF-IDF baseline and a majority floor.

**11. Why do I believe it may work?**
IndicXNLI: MuRIL beats XLM-R/mBERT/IndicBERT on Indic languages and English + Indic training works best. The
`claim [SEP] comment` input lets the model attend to the claim. I do NOT claim that more English data helps: in
CASE@EACL 2024 HAMiSoN-Ensemble, external data worsened RoBERTa and Llama 2, so the English training mix is an ablation
in Milestone 2. I will not promise a number: the peer evidence says test scores can be far below validation.

**12. What have I implemented?**
Loader with validation gate (`data.py`), preprocessing + script ID (`preprocess.py`), EDA (`eda.py`), TF-IDF baselines
(`baseline.py`), transformer fine-tuning and predictor (`train_transformer.py`, `predict.py`: written, **not trained**),
external-data fetcher, a 150-row synthetic pilot corpus, and each module self-checks with asserts.

**13. What did I learn from the initial experiments?** (English merged, 5-fold CV, macro-F1)

| setting | macro-F1 |
|---|---|
| majority-class floor | **0.200** |
| word 1-2 grams, LogReg | 0.588 |
| char_wb 2-5, LogReg | 0.605 |
| word + char, LogReg (best) | **0.613 +- 0.025** |
| word + char, LinearSVC | 0.590 |
| best config, no class weights | 0.558 |

* Char n-grams beat word n-grams, and combining is best: modest but consistent.
* Class weighting: macro-F1 0.558 -> 0.613 and **Against recall 0.25 -> 0.54**.
* Biggest error: **None predicted as Favour (286 sentences)**: factual climate statements ("sea level rise is predicted to
  threaten Florida") have no opinion marker but sound pro-concern. This is the implicit-stance problem in miniature.
* 69 Against sentences were called None.
* **Pilot results are development only.** Pilot CV 0.57-0.58; English -> pilot 0.341 (floor 0.167).

---

## B. Likely follow-ups

**Why MuRIL over XLM-R?**
IndicXNLI (Aggarwal et al. 2022) reports MuRIL consistently ahead of XLM-R, mBERT and IndicBERT across Indic languages;
XLM-R wins only in the English-evaluation setup. The authors attribute MuRIL's advantage to model size, Indic-specific
pretraining data, an MLM + TLM objective mix and transliterated pretraining data; the MuRIL paper itself says it uses
translated and transliterated document pairs and handles Latin-script test data, which is relevant to Romanised text.
Caveat I should volunteer: that is evidence from an NLI task, not stance. I keep XLM-R as a flagged comparison. (A competing team's XLM-R run scored
above their MuRIL runs, but their README says the runs differ in split, seed and learning rate, so it is not a controlled
comparison.)

**Why not IndicBERT?**
Same paper: it underperforms even mBERT, attributed to its small size. It stays selectable as a flag for completeness.

**Why macro-F1 and not accuracy?**
Accuracy rewards predicting the big classes. The majority-class floor has 42.9% accuracy but macro-F1 of **0.200**.
Macro-F1 averages per-class F1 with equal weight, so the rare Against class (14%) counts as much as the others, and
the task asks for exactly that.

**Why class weights?**
Same model, only the weights changed: macro-F1 0.558 -> 0.613, Against recall 0.25 -> 0.54. Unweighted training minimises
overall error, which is dominated by Favour and None.

**Why character n-grams?**
Measured hapax rate: 54% (English), 68% (pilot). Most word types are seen once, so word features have no statistics.
`char_wb` 2-5 grams share sub-strings across spellings (`jhooth/jhuth`, `sarkar/sarkaar`) and work on Romanised and
code-mixed text without a tokeniser. Measured: char 0.605 vs word 0.588, combined 0.613.

**How will you handle code-mixing?**
(1) Subword encoders pretrained on Indic + English (MuRIL), (2) char n-grams in the baseline, (3) script ID to route and
to slice evaluation, (4) optional transliteration of Romanised text to native script, (5) unlabelled domain-adaptive
pretraining on the Swami corpus. The pilot has 22 code-mixed and 22 Romanised rows to check this.

**How will you handle implicit stance?**
It cannot be solved by keywords. The pair input gives the model the claim; pretrained encoders bring world knowledge
(drought, glaciers, floods); I will report accuracy on the pilot's `implicit` slice separately instead of hiding it in an
average. Honest expectation: this will remain the weakest slice.

**Why can't the demonetisation corpus be used as climate training data?**
Different target. Its FAVOR means "favours demonetisation". Training on it would teach "praises a government policy" as
climate-Favour. It is a textbook cross-target mismatch, kept in a separate file for methodology and unlabelled pretraining.

**Why not just use the Dhoroni data for Bengali?**
It is Bengali, and its Positive label ("supports climate change as a real and pressing issue") is close to our claim, but
it is long news articles (median 331 words) and Neutral/Negative are 21% and 6%. Domain gap; it is a transfer and
domain-adaptation candidate, not comments.

**Why not use ClimaConvo?**
15,309 rows but IDs and labels only; text requires the paid X API.

**What is translate-train vs zero-shot?**
Zero-shot: fine-tune on English, test directly on Hindi/Bengali (no translation noise, no in-language supervision).
Translate-train: machine-translate English training data to Hindi/Bengali and train on it (in-language supervision, but
translation noise and loss of social-media register). Plan: compare both and their union.

**Your pilot set is synthetic. Why trust anything on it?**
I don't use it for performance claims. It checks that the pipeline handles Devanagari, Bengali, Romanised and
code-mixed text, and organises error analysis by phenomenon. It was written by the team (drafted with LLM assistance)
because the official data was not in hand; numbers on it are labelled development only.

**Aren't English CV scores optimistic?**
Yes, and I say so: no document ids (sentences from one article may straddle folds), 21 tweets share one hashtag, and the
set mixes news and tweets. A competing team's validation was 0.65-0.68 vs official 0.42-0.53.

**Didn't CASE 2024 show that external data helps?**
No, the opposite for the parts I checked. HAMiSoN-Ensemble added external data (SemEval climate to RoBERTa, all SemEval
topics to Llama 2, OLID to the multi-task model): performance improved only for the multi-task model and worsened for
RoBERTa and Llama 2; they concluded it "did not improve the performance". So the extra-English-data question is open, and
I treat it as an ablation, not an assumption.

**Why drop the 202 tie rows rather than assign them a label?**
No annotator majority means no defensible gold label (215 of 2,300 raw GWSD sentences, 9.3%; 202 survive our deduplication). They are kept in a separate file and reported as evidence of
ambiguity, not as missing data.

**What is the loader gotcha?**
`pandas.read_csv` converts the literal string "None" to NaN. One of our classes is "None", so a naive read silently deletes
a third of the data. Fix: `keep_default_na=False`; regression assert in `src/data.py`.

**Did you use another team's work?**
We found a competing AISoMe team's repository and used **only the scores it reports**, cited by name. No code, data
or methodology was copied.

**What is the single biggest risk?**
The gap between validation and the real hidden test; mitigation is evaluating per phenomenon, never tuning on the
pilot, and reporting the zero-shot baseline alongside.
