# Literature review: stance detection on global warming for Indic languages

**Project:** Detecting Stance in Media on Global Warming for Indic Languages (CSET346, Milestone 1)
**Claim:** "Climate change and global warming is a serious concern." **Labels:** Favour / Against / None.

**Origin of the project.** The task is derived from the **AISoMe 2026 shared task at FIRE 2026 (Kolkata)**: same claim, same
three labels, same two languages. Its defining property is that **publicly available training data is in English while the
test data is Hindi and Bengali YouTube comments.** The problem is therefore **cross-lingual transfer**, not plain
three-class text classification.

**Reference status.** Every reference below was checked on 13 Oct 2026 against its primary page (arXiv abstract page, ACL
Anthology record, or the paper's full text); the exact depth is in the table at the end. Claims that a source could not
support were removed or reworded. Numbers in this review come from those sources or from our own measured runs (`outputs/`).

---

## 1. Stance detection fundamentals

Stance detection asks whether the author of a text is in favour of, against, or neutral toward a **target** that may never
be mentioned. It differs from sentiment analysis: "The ban is a disaster" is negative in sentiment and may be in favour of
a liberal policy. AlDayel and Magdy (2021) survey stance detection on social media: task definition, target types,
features, learning methods and benchmark results.

The three-way formulation we use comes from **SemEval-2016 Task 6** (Mohammad et al., 2016): tweets labelled
FAVOR / AGAINST / NONE toward a given target. Per the HAMiSoN-Ensemble paper's description of that dataset, it has 4,163
tweets over five topics (abortion, climate, Hillary Clinton, feminism, atheism); the climate target,
*Climate Change is a Real Concern*, is the origin of our label set. We measured that subset: 564 tweets, of which only
26 are Against (4.6%).

Two structural facts matter for everything below:

* **Target-dependence.** The same sentence has different stance toward different targets. Models that ignore the target
  can only exploit topic words.
* **Cross-target generalisation.** Applying a model trained on some targets to unseen targets is a recognised hard
  setting; Khiabani and Zubiaga (2024) survey the techniques (attention, adversarial learning, fine-tuning for robustness,
  prompt-tuning, external knowledge) and datasets. Our Swami corpus (section 4) is a concrete example of why labels with the
  same names must not be pooled across targets.

## 2. Climate-change stance

| Resource | What it is | Notes |
|---|---|---|
| **GWSD / GWStance** (Luo, Card & Jurafsky, 2020, *Detecting Stance in Media On Global Warming*, Findings of EMNLP 2020, pp. 3296-3315) | News-media sentences with 8 crowd annotations each; the released `GWSD.tsv` has 2,300 rows, per-annotator labels (agrees / disagrees / neutral), model-estimated label probabilities and a 200-sentence held-out flag | The target string recorded in the team's files is "global warming is a serious concern"; the paper's abstract uses that phrase in an example sentence, and we did not confirm it as the formal annotation prompt. **Measured by us on the raw file: 215 of 2,300 sentences (9.3%) have no annotator majority**; 202 survive deduplication in the merged file |
| **ClimaConvo** (Shiwakoti, Thapa, Rauniyar, Shah, Bhandari & Naseem, 2024, LREC-COLING 2024, pp. 984-994) | 15,309 climate tweets; annotations for relevance, stance, hate speech, direction of hate, targets and humour | **The public repository ships tweet IDs and labels only, no text** (needs the paid X API). Stance values in the CSV: support 6,146, denial 994, neutral 3,267, among 10,407 relevant tweets (counted by us) |
| **CASE@EACL 2024 ClimateActivism shared task** (Thapa et al., 2024, CASE 2024, pp. 234-247) | Hate-speech detection and target classification (subtasks A, B) and stance classification (subtask C) on climate-activism tweets; ranked by macro-F1 | Over 100 teams registered; top reported scores 91.44% (subtask C), 78.58% (B), 74.83% (A); summaries of 24 leaderboard teams |
| **HAMiSoN-Ensemble** (Rodriguez-Garcia, Reyes-Montesinos, Fraile-Hernandez & Penas, 2024, CASE 2024, pp. 118-124) | Voting ensemble of **RoBERTa + Llama 2 + multi-task learning** for the stance subtask | Studied adding external data (SemEval-2016 climate for RoBERTa; all SemEval topics for Llama 2; OLID for the multi-task model). **Result: performance improved only for the multi-task model and worsened for RoBERTa and Llama 2; the authors conclude this augmentation "did not improve the performance"** and call for further analysis. The best submitted ensemble was second on the leaderboard; a post-competition ensemble of the best models would have been first |
| **MultiClimate** (Wang, Zuo, Peng & Plank, 2024; arXiv:2409.18346) | 100 YouTube climate videos, 4,209 frame-transcript pairs, support / neutral / oppose; inter-annotator kappa 0.703 | Best fusion model 0.747 accuracy / 0.749 F1; text-only BERT beats image-only ResNet50 and ViT; ~100M-parameter fusion beats 9B IDEFICS, Llama 3 and Gemma 2. English; transcript sentences, not comments. "Oppose" = negative sentiment toward climate change itself |
| **Dhoroni** (Wasi, Faisal, Ahmad, Rahman & Islam, 2024; arXiv:2410.17225) | 2,300 Bangla climate/environment news articles with ten annotation perspectives including stance | Positive = supports climate change as a real, pressing issue. BanglaBERT-Dhoroni stance baseline: **accuracy 0.635, F1 0.308**: strong on the majority class, weak on minority stances |

Take-aways for us: (i) the target-correct resources are English; (ii) the few Indic ones are news, not comments;
(iii) minority stances (Against, Negative) are consistently the weak point; (iv) **whether adding more English data
helps is an open empirical question**, not an established result (HAMiSoN-Ensemble found it did not for RoBERTa).

## 3. Multilingual and cross-lingual stance

Multilingual encoders (mBERT; XLM-R, Conneau et al., 2020) are pretrained on 100+ languages and enable zero-shot transfer
from English fine-tuning. Two limits are relevant here:

* Transfer is weakest for low-resource languages, non-Latin scripts and **Romanised or code-mixed** text, which pretraining
  sees little of. Our own measurement (section 6): a TF-IDF model trained on English predicts "None" for 131 of 150 Indic
  pilot comments.
* Social-media stance depends on register (sarcasm, slang) that machine translation (the alternative bridge) tends to
  flatten.

Analogue: **StanceEval** (Arabic stance detection; the 2024 edition was held at ArabicNLP 2024, the 2026 edition uses the
Mawqif-v2 tweets labelled Favor / Against / None). Its task description states that the target "may be abstract or not
explicitly mentioned in the text", and the 2026 edition adds tracks for seen and unseen targets, the same direction
(generalisation across targets) as AISoMe.

**Closest prior art: AISoMe 2026 participants.** While searching for datasets we found a competing team's public repository
(`Debajyoti212/Aisome2026`, team Squad4). **We deliberately limited our use of it to the scores it reports**, which we
cite below by name; we did not use its code, data pipeline or methodology. Squad4 used translate-train with MuRIL and
XLM-R; their README reports official macro-F1 of 0.4222-0.4561 (Hindi, rank 17 of 18) and 0.4699-0.5274 (Bengali, rank 12
of 18), against internal validation of 0.6533-0.6817. These are participant-reported, not peer-reviewed, and not
independently verified. The size of that **validation-to-test gap** is the most useful single fact for planning:
internal cross-validation on English or translated data overstates performance on real Hindi/Bengali comments.

## 4. Hindi, Bengali and code-mixed NLP

**Encoders.**

* **MuRIL** (Khanuja et al., 2021): argues that existing multilingual models give Indian languages little representation
  in their vocabularies and training data; trained on Indian-language text, with translated and transliterated document
  pairs as cross-lingual supervision; outperforms multilingual BERT on all XTREME tasks and handles transliterated
  (Latin-script) test data. IndicXNLI describes it as trained on 17 Indic languages plus English.
* **IndicBERT** (Kakwani et al., 2020, IndicNLPSuite): the Indic model released with the IndicNLPSuite benchmarks and
  IndicCorp; IndicXNLI lists it at 33M parameters, pretrained on 11 Indic languages plus English.
* **XLM-R** (Conneau et al., 2020): cross-lingual RoBERTa-style model pretrained on 100+ languages.
* **IndicXNLI** (Aggarwal, Gupta & Kunchukuttan, 2022, EMNLP 2022, pp. 10994-11006): an NLI dataset for 11 Indic
  languages made by machine-translating XNLI. Findings from its results analysis (on NLI, not stance): **MuRIL performs the
  best across all Indic languages except in the English-evaluation setup**; XLM-R beats MuRIL in rare cases that deal solely
  with English data; **IndicBERT performs worse than mBERT**, which the authors attribute to its smaller size (33M vs 167M
  parameters); and the **English + Indic train strategy outperforms** the others, with MuRIL best. The authors attribute
  MuRIL's advantage to model size, Indic-specific pretraining data, a mixture of MLM and TLM objectives, and transliterated
  pretraining data. This is our empirical justification for MuRIL as the primary encoder, with the caveat that it is
  evidence from another task.

**Code-mixed stance.** Swami, Khandelwal, Singh, Akhtar and Shrivastava (2018), *An English-Hindi Code-Mixed Corpus: Stance
Annotation and Baseline System* (arXiv:1805.11868): 3,545 tweets on demonetisation (964 favour / 647 against / 1,934 none,
Cohen's kappa 0.82), token-level language tags, and a baseline of **58.7% accuracy** (10-fold CV; RBF-kernel SVM with all
features; character n-grams alone 55.4%, word n-grams 54.6%). It is the only Hindi code-mixed stance resource we found. Its
target is demonetisation, so it informs **methodology only**.

**Indic-specific preprocessing findings from our own work** (all asserted in `src/preprocess.py`):

1. Nukta letters have two encodings (U+0958 vs U+0915 U+093C); NFC must run first or visually identical strings differ.
2. Repetition collapse acts on code points: `नहीीीीी` becomes `नहीी` because the vowel sign is its own code point.
3. Python's `\w` and scikit-learn's default token pattern split Hindi words at vowel signs (`जलवायु` -> `जलव`, `य`), so an
   Indic-aware token pattern is required for any word-level feature.
4. Script ID from Unicode ranges separates Devanagari from Bengali but cannot separate Romanised Hindi from English.

## 5. LLM-based stance detection

Pangtey, Bhatnagar, Bansal, Dar and Kumar (2025), *Large Language Models Meet Stance Detection: A Survey of Tasks, Methods,
Applications, Challenges and Future Directions* (arXiv:2505.08464), propose a taxonomy along learning methods (supervised,
unsupervised, few-shot, zero-shot), data modalities (unimodal, multimodal, hybrid) and target relationships (in-target,
cross-target, multi-target); they list implicit stance expression, cultural bias and computational constraints among the
open challenges. Cruickshank and Ng (2023, arXiv:2309.13734) evaluated 10 open-source LLMs with 7 prompting schemes for
stance classification and report that LLMs are competitive with in-domain supervised models but not consistently so, and
that "fine-tuning process does not necessarily lead to better performance". Caveat for our methodology: prompting a
LLM zero-shot is a legitimate baseline and should be run in Milestone 2 before claiming that fine-tuning helps. LLM
coverage of Bengali and Romanised Bengali is an expectation to test, not a measured fact here.

## 6. State of the art at a glance

Numbers are as reported by the source, on that source's own data. **They are not comparable with each other or with our
results.** Our own measurements are marked "measured".

| Approach family | Representative work | Reported result | Limitation for this project |
|---|---|---|---|
| Classical features on code-mixed Hindi | Swami et al., 2018 | 58.7% accuracy (10-fold CV, RBF-SVM, all features) | Target is demonetisation; Romanised Hindi |
| TF-IDF word + char n-grams, class-weighted LogReg (**ours, measured**) | this repo | macro-F1 **0.613** (5-fold CV, English merged 2,408) vs majority floor **0.200**; **0.341** zero-shot on the Indic pilot (dev only, not comparable) | English CV is optimistic; no cross-lingual capacity |
| Shared-task systems, English climate-activism tweets | CASE 2024 (Thapa et al.) | top subtask C macro-F1 91.44% | English; in-domain training data |
| RoBERTa + Llama 2 + multi-task voting ensemble | HAMiSoN-Ensemble (Rodriguez-Garcia et al., 2024) | second on the stance leaderboard as submitted (first in post-competition analysis); external data worsened RoBERTa and Llama 2 | English activism tweets; augmentation not shown to help |
| Multimodal fusion | Wang et al., 2024 (MultiClimate) | 0.747 accuracy / 0.749 F1 | English video transcripts; not comments |
| Language-specific encoder | Wasi et al., 2024 (BanglaBERT-Dhoroni) | accuracy 0.635, F1 0.308 | Bengali **news**; minority stances poorly recovered |
| Translate-train + MuRIL / XLM-R (same shared task) | Squad4, AISoMe 2026 (participant report) | official macro-F1 0.4222-0.4561 (Hindi), 0.4699-0.5274 (Bengali); internal validation 0.65-0.68 | Large validation-to-test gap; participant-reported |
| LLM prompting vs fine-tuning for stance | Cruickshank & Ng, 2023 | LLMs competitive with in-domain supervised models; fine-tuning not necessarily better | Indic and code-mixed coverage uncertain |
| News-sentence stance (GWSD), climate tweets (ClimaConvo) | Luo et al., 2020; Shiwakoti et al., 2024 | see the papers (not reproduced here) | English; news not comments; ClimaConvo text not distributable |

## 7. Research gap

1. **Language and register gap.** Target-correct stance data (GWSD, SemEval-2016) is English. No public Hindi or Bengali
   corpus of social-media comments is labelled for stance on this claim; Dhoroni is Bengali news and Swami is Romanised Hindi
   on a different target. (Evidence: `data/DATASET_INVENTORY.md`.)
2. **Resource availability.** The largest climate-tweet corpus, ClimaConvo, cannot be used as distributed (IDs only).
3. **Transfer is under-evaluated.** The only shared-task evidence we found shows a large drop from internal validation to
   the real Hindi/Bengali test (0.65-0.68 to 0.42-0.53). Little is known about which test phenomena (negation, sarcasm,
   implicit stance, code-mixing, Romanisation) cause it. Our own zero-shot probe (English-trained TF-IDF on the pilot)
   collapses to majority-class prediction.
4. **Minority-class behaviour.** In every dataset we measured or read, "against" is rare (4.6% of SemEval climate, 14.3%
   of our English training set; Dhoroni "negative" 6%) and is the class models recover worst, yet macro-F1 weights it equally.
5. **The value of extra English data is unsettled.** HAMiSoN-Ensemble reports that adding external stance data did not
   improve RoBERTa or Llama 2 on the CASE stance task, so "more English data" cannot be assumed to help.

## 8. Proposed methodology

Staged so that every stage is explainable and measurable:

1. **Normalise** (NFC, strip URLs/mentions/timestamps/zero-width characters, collapse 3+ repeats per code point, keep emoji)
   and identify script (Devanagari / Bengali / Latin / mixed) from Unicode ranges.
2. **Floor and baselines.** Majority-class floor; TF-IDF word 1-2 + char_wb 2-5 with LogReg / LinearSVC, class-weighted.
   *Done in Milestone 1.*
3. **Target-aware multilingual encoder.** Fine-tune MuRIL (XLM-R and IndicBERT as comparisons) on the pair
   `claim [SEP] comment`, class-weighted loss, macro-F1 for model selection, fixed seed. *Script written, not yet run.*
4. **Cross-lingual strategy.** Compare zero-shot transfer from English, translate-train (IndicTrans2/NLLB), and English +
   translated data (IndicXNLI's best strategy was English + Indic). Treat the English training mix as an ablation
   (SemEval-only vs GWSD + SemEval), since the HAMiSoN-Ensemble result shows extra data can hurt. *Planned.*
5. **Evaluation.** Macro-F1 primary; accuracy, per-class precision/recall and confusion matrix supporting; per-phenomenon and
   per-script breakdown; a zero-shot LLM baseline given the Cruickshank and Ng caveat. The pilot corpus is for development
   only; no number from it is reported as an official result.

---

## References and verification status

"Page" = bibliographic record or abstract page opened and checked. "Full text" = the paper body was read for the specific
claim used here. Items marked * are not peer-reviewed.

| # | Reference | Status |
|---|---|---|
| 1 | Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., Cherry, C. (2016). SemEval-2016 Task 6: Detecting Stance in Tweets. *SemEval-2016*, pp. 31-41. aclanthology.org/S16-1003 | Page verified (title, authors, venue, pages). Dataset facts from HAMiSoN-Ensemble full text and our own data |
| 2 | Luo, Y., Card, D., Jurafsky, D. (2020). Detecting Stance in Media On Global Warming. *Findings of EMNLP 2020*, pp. 3296-3315. aclanthology.org/2020.findings-emnlp.296 | Page verified; citation also as stated in the GWStance repository README; data file inspected. No LICENSE file in the repository |
| 3 | Shiwakoti, S., Thapa, S., Rauniyar, K., Shah, A., Bhandari, A., Naseem, U. (2024). Analyzing the Dynamics of Climate Change Discourse on Twitter: A New Annotated Corpus and Multi-Aspect Classification. *LREC-COLING 2024*, pp. 984-994. aclanthology.org/2024.lrec-main.88 | Page verified; repository CSV inspected |
| 4 | Thapa, S., Rauniyar, K., Jafri, F., Shiwakoti, S., Veeramani, H., Jain, R., Kohli, G. S., Hurriyetoglu, A., Naseem, U. (2024). Stance and Hate Event Detection in Tweets Related to Climate Activism - Shared Task at CASE 2024. *CASE 2024*, pp. 234-247. aclanthology.org/2024.case-1.33 | Page verified (abstract-level facts only) |
| 5 | Rodriguez-Garcia, R., Reyes-Montesinos, J., Fraile-Hernandez, J. M., Penas, A. (2024). HAMiSoN-Ensemble at ClimateActivism 2024: Ensemble of RoBERTa, Llama 2, and Multi-task for Stance Detection. *CASE 2024*, pp. 118-124. aclanthology.org/2024.case-1.16 | Full text read; the external-data finding is quoted from its discussion |
| 6 | Wang, J., Zuo, L., Peng, S., Plank, B. (2024). MultiClimate: Multimodal Stance Detection on Climate Change Videos. arXiv:2409.18346 | Page and HTML text read |
| 7 | Wasi, A. T., Faisal, W., Ahmad, T., Rahman, A., Islam, M. R. (2024). Dhoroni: Exploring Bengali Climate Change and Environmental Views with a Multi-Perspective News Dataset and Natural Language Processing. arXiv:2410.17225 | Dataset card and paper text read |
| 8 | Swami, S., Khandelwal, A., Singh, V., Akhtar, S. S., Shrivastava, M. (2018). An English-Hindi Code-Mixed Corpus: Stance Annotation and Baseline System. arXiv:1805.11868 | Full text read (counts, kappa, 58.7% in Table 4); counts re-verified on the released data |
| 9 | Khanuja, S. et al. (2021). MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730 | Abstract page verified |
| 10 | Kakwani, D., Kunchukuttan, A., Golla, S., N.C., G., Bhattacharyya, A., Khapra, M. M., Kumar, P. (2020). IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages. *Findings of EMNLP 2020*, pp. 4948-4961. aclanthology.org/2020.findings-emnlp.445 | Page verified |
| 11 | Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*, pp. 8440-8451. aclanthology.org/2020.acl-main.747 | Page verified |
| 12 | Aggarwal, D., Gupta, V., Kunchukuttan, A. (2022). IndicXNLI: Evaluating Multilingual Inference for Indian Languages. *EMNLP 2022*, pp. 10994-11006; arXiv:2204.08776 | arXiv page and full text read; EMNLP 2022 proceedings entry located via ACL Anthology search result |
| 13 | Pangtey, L., Bhatnagar, A., Bansal, S., Dar, S. S., Kumar, N. (2025). Large Language Models Meet Stance Detection: A Survey of Tasks, Methods, Applications, Challenges and Future Directions. arXiv:2505.08464 | Abstract page verified |
| 14 | Khiabani, P. J., Zubiaga, A. (2024). Cross-Target Stance Detection: A Survey of Techniques, Datasets, and Challenges. arXiv:2409.13594 | Abstract page verified |
| 15 | Cruickshank, I. J., Ng, L. H. X. (2023). Prompting and Fine-Tuning Open-Sourced Large Language Models for Stance Classification. arXiv:2309.13734 | Abstract page verified |
| 16 | AlDayel, A., Magdy, W. (2021). Stance Detection on Social Media: State of the Art and Trends. *Information Processing & Management* 58(4), 102597; arXiv:2006.03644 | arXiv page verified |
| 17 | StanceEval 2024 / 2026 (Arabic stance detection shared tasks). stanceeval.github.io | Website read; only what the page states is used |
| 18* | Squad4 (2026). Translate-Train Climate-Change Stance Detection in Hindi and Bengali, AISoMe 2026 at FIRE 2026 (repository README). github.com/Debajyoti212/Aisome2026 | README read; only reported scores cited; competing team, no code/data/method reused |
| 19* | AISoMe 2026 track at FIRE 2026, Kolkata (Hindi/Bengali YouTube comments, Favour/Against/None, English training data) | Task description from the course brief; corroborated by a web-search summary; the official track page itself was not opened |

Removed because it could not be verified: Küçük, D. & Can, F. (2020), "Stance Detection: A Survey", ACM Computing Surveys.
Removed claims: that MuRIL "allocates far more of its vocabulary to Indic scripts" (no source checked says this); that
StanceEval documents dialectal and spelling variation (its page does not); that external SemEval-2016 data helped stance
detection in CASE 2024 (HAMiSoN-Ensemble found the opposite for RoBERTa and Llama 2).
