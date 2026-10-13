# Literature review: stance detection on global warming for Indic languages

**Project:** Detecting Stance in Media on Global Warming for Indic Languages (CSET346, Milestone 1)
**Claim:** "Climate change and global warming is a serious concern." **Labels:** Favour / Against / None.

**Origin of the project.** The task is derived from the **AISoMe 2026 shared task at FIRE 2026 (Kolkata)**: same claim, same
three labels, same two languages. Its defining property is that **publicly available training data is in English while the
test data is Hindi and Bengali YouTube comments.** The problem is therefore **cross-lingual transfer**, not plain
three-class text classification.

**Reference status.** References marked **[read]** had their abstract or key facts opened during this project.
References marked **[recall]** are cited from the team's knowledge or the task brief; check authors, pages and numbers
against the source before submission. Numbers in this review come only from sources marked [read] or from our own
measured runs (`outputs/`).

---

## 1. Stance detection fundamentals

Stance detection asks whether the author of a text is in favour of, against, or neutral toward a **target** that may never
be mentioned. It differs from sentiment analysis: "The ban is a disaster" is negative in sentiment and may be in favour of
a liberal policy. Küçük and Can (2020) survey the field [recall].

The formulation we use comes from **SemEval-2016 Task 6** (Mohammad et al., 2016) [read via data]: tweets labelled
FAVOR / AGAINST / NONE toward a given target, where NONE covers both neutral and no-stance. Its target list includes
*Climate Change is a Real Concern*, the origin of our label set. We measured that subset: 564 tweets, of which only
26 are Against (4.6%).

Two structural facts matter for everything below:

* **Target-dependence.** The same sentence has different stance toward different targets. Models that ignore the target
  can only exploit topic words.
* **Cross-target generalisation.** Training on one target and testing on another is a recognised hard setting; surveys of
  cross-target stance detection (arXiv 2409.13594 [recall]) catalogue the approaches. Our Swami corpus (section 4) is a
  concrete example of why labels with the same names must not be pooled across targets.

## 2. Climate-change stance

| Resource | What it is | Notes |
|---|---|---|
| **GWStance / GWSD** (Luo, Card & Jurafsky, 2020, *Detecting Stance in Media on Global Warming*, Findings of EMNLP) [recall] | News-media sentences annotated for stance toward "global warming is a serious concern" | Annotation target is our claim verbatim. In the team's file: 2,047 sentences, 8 annotators each, majority vote; **202 (9.9%) had no majority** |
| **ClimaConvo** (Shiwakoti et al., 2024, LREC-COLING) [read: README/CSV] | 15,309 climate tweets, six annotation layers (relevance, stance, hate, direction, target, humour) | **The public repository ships tweet IDs and labels only, no text** (needs the paid X API). Stance values in the CSV: support 6,146, denial 994, neutral 3,267, among 10,407 relevant tweets |
| **CASE@EACL 2024 ClimateActivism** (Thapa et al., 2024) [recall] | Shared task: stance and hate-event detection in tweets about climate activism | HAMiSoN-Ensemble combined **RoBERTa + Llama 2 + multi-task learning**; HAMiSoN's baselines showed that **external SemEval-2016 data helped the stance subtask**, direct evidence for our external-data strategy |
| **MultiClimate** (Wang, Zuo, Peng & Plank, 2024; arXiv 2409.18346) [read] | 100 YouTube climate videos, 4,209 frame-transcript pairs, support / neutral / oppose; inter-annotator kappa 0.703 | Best reported fusion model 0.747 accuracy / 0.749 F1; text-only BERT beats image-only ResNet50 and ViT; ~100M-parameter fusion beats 9B IDEFICS, Llama 3 and Gemma 2. English; transcript sentences, not comments |
| **Dhoroni** (Wasi et al., 2024; arXiv 2410.17225) [read] | 2,300 Bangla climate/environment news articles with ten annotation perspectives including stance | Positive = supports climate change as a real, pressing issue. BanglaBERT-Dhoroni stance baseline: **accuracy 0.635, F1 0.308**: strong on the majority class, weak on minority stances |

Take-aways for us: (i) the target-correct resources are English; (ii) the few Indic ones are news, not comments;
(iii) minority stances (Against, Negative) are consistently the weak point.

## 3. Multilingual and cross-lingual stance

Multilingual encoders (mBERT, XLM-R: Conneau et al., 2020 [recall]) align languages through shared sub-word vocabularies and
joint pretraining, enabling zero-shot transfer from English fine-tuning. Two limits are relevant here:

* Transfer is weakest for low-resource languages, non-Latin scripts and **Romanised or code-mixed** text, which pretraining
  sees little of.
* Social-media stance depends on register (sarcasm, slang) that machine translation (the alternative bridge) tends to
  flatten.

Analogue: the **StanceEval** shared tasks for Arabic (2024; 2026) [recall] show that dialectal variation, informal
expressions, spelling variation and implicit language make non-English stance detection hard, and the 2026 edition moved
to cross-target generalisation, the same direction as AISoMe.

**Closest prior art: AISoMe 2026 participants.** While searching for datasets we found a competing team's public repository
(`Debajyoti212/Aisome2026`, team Squad4). **We deliberately limited our use of it to the scores it reports**, which we
cite below by name; we did not use its code, data pipeline or methodology. Squad4 used translate-train with MuRIL and
XLM-R; their README reports official macro-F1 of 0.4222-0.4561 (Hindi, rank 17 of 18) and 0.4699-0.5274 (Bengali, rank 12
of 18), against internal validation of 0.6533-0.6817 [read; participant-reported, not independently verified]. The size of
that **validation-to-test gap** is the most useful single fact for planning: internal cross-validation on English or
translated data overstates performance on real Hindi/Bengali comments.

## 4. Hindi, Bengali and code-mixed NLP

**Encoders.**

* **MuRIL** (Khanuja et al., 2021) [recall]: BERT pretrained on 17 Indian languages plus English, including translated and
  transliterated text pairs, which targets exactly Romanised and code-mixed input.
* **IndicBERT** (Kakwani et al., 2020, IndicNLPSuite) [recall]: a small ALBERT-style model pretrained on Indic corpora.
* **XLM-R** (Conneau et al., 2020) [recall]: 100-language RoBERTa-style model; strong, English-centric allocation.
* **IndicXNLI** (Aggarwal et al., 2022; arXiv 2204.08776) [recall; findings as given in the task brief]: across Indic
  languages **MuRIL consistently beats XLM-R, mBERT and IndicBERT**; XLM-R wins only on English-centric data;
  **IndicBERT underperforms even mBERT**, attributed to its smaller size; and **English + Indic combined training works
  best**. MuRIL also allocates much more of its vocabulary to Indic scripts, so native words tokenise into meaningful
  sub-words rather than fragmenting. This is our empirical justification for MuRIL as the primary encoder.

**Code-mixed stance.** Swami et al. (2018), *An English-Hindi Code-Mixed Corpus: Stance Annotation and Baseline System*
(arXiv 1805.11868) [read]: 3,545 tweets on demonetisation (964 favour / 647 against / 1,934 none, Cohen's kappa 0.82),
token-level language tags, and a classical-feature baseline of **58.7% accuracy** (10-fold CV). It is the only Hindi
code-mixed stance resource we found. Its target is demonetisation, so it informs **methodology only**.

**Indic-specific preprocessing findings from our own work** (all asserted in `src/preprocess.py`):

1. Nukta letters have two encodings (U+0958 vs U+0915 U+093C); NFC must run first or visually identical strings differ.
2. Repetition collapse acts on code points: `नहीीीीी` becomes `नहीी` because the vowel sign is its own code point.
3. Python's `\w` and scikit-learn's default token pattern split Hindi words at vowel signs (`जलवायु` -> `जलव`, `य`), so an
   Indic-aware token pattern is required for any word-level feature.
4. Script ID from Unicode ranges separates Devanagari from Bengali but cannot separate Romanised Hindi from English.

## 5. LLM-based stance detection

Pangtey et al. (2025), *Large Language Models Meet Stance Detection: A Survey* (arXiv 2505.08464) [recall], organise the
area by learning method (fine-tuning, prompting, in-context learning), modality, and in-/cross-/multi-target settings.
Cruickshank and Ng (arXiv 2309.13734) [recall] compare prompting with fine-tuning open LLMs for stance classification and
find that **fine-tuned models do not necessarily beat zero-shot prompting**. Caveat for our methodology: a zero-shot LLM
is a legitimate baseline and should be run in Milestone 2 before claiming that fine-tuning helps. LLM coverage of Bengali
and especially Romanised Bengali is weaker than of English; this is an expectation to test, not a measured fact here.

## 6. State of the art at a glance

Numbers are as reported by the source, on that source's own data. **They are not comparable with each other or with our
results.** Our own measurements are marked "measured".

| Approach family | Representative work | Reported result | Limitation for this project |
|---|---|---|---|
| Hand-crafted n-gram + linear (SemEval-2016 era) | Mohammad et al., 2016 | see paper | English tweets; target-specific; no Indic |
| TF-IDF word + char n-grams, class-weighted LogReg (**ours, measured**) | this repo | macro-F1 **0.613** (5-fold CV, English merged 2,408) vs majority floor **0.200**; **0.341** zero-shot on the Indic pilot (dev only, not comparable) | English CV is optimistic; no cross-lingual capacity |
| Fine-tuned encoder on news sentences | Luo et al., 2020 (GWSD) | see paper | English news, not comments |
| Fine-tuned encoders on climate tweets | Shiwakoti et al., 2024 (ClimaConvo) | see paper | English; text not distributable |
| Encoder + LLM ensemble, multi-task | HAMiSoN, CASE@EACL 2024 (Thapa et al.) | see task report | English activism tweets; reported finding: external SemEval data helped |
| Multimodal fusion | Wang et al., 2024 (MultiClimate) | 0.747 accuracy / 0.749 F1 | English video transcripts; not comments |
| Language-specific encoder | Wasi et al., 2024 (BanglaBERT-Dhoroni) | accuracy 0.635, F1 0.308 | Bengali **news**; minority stances poorly recovered |
| Classical features on code-mixed Hindi | Swami et al., 2018 | 58.7% accuracy (10-fold CV) | Target is demonetisation; Romanised Hindi |
| Translate-train + MuRIL / XLM-R (same shared task) | Squad4, AISoMe 2026 (participant report) | official macro-F1 0.4222-0.4561 (Hindi), 0.4699-0.5274 (Bengali); internal validation 0.65-0.68 | Large validation-to-test gap; participant-reported |
| Zero-shot / few-shot LLM prompting | Cruickshank & Ng; Pangtey et al. | fine-tuned not necessarily better | Indic and code-mixed coverage uncertain |

## 7. Research gap

1. **Language and register gap.** Target-correct stance data (GWSD, SemEval-2016) is English. No public Hindi or Bengali
   corpus of social-media comments is labelled for stance on this claim; Dhoroni is Bengali news and Swami is Romanised Hindi
   on a different target. (Evidence: `data/DATASET_INVENTORY.md`.)
2. **Resource availability.** The largest climate-tweet corpus, ClimaConvo, cannot be used as distributed (IDs only).
3. **Transfer is under-evaluated.** The only shared-task evidence we found shows a large drop from internal validation to
   the real Hindi/Bengali test (0.65-0.68 to 0.42-0.53). Little is known about which test phenomena (negation, sarcasm,
   implicit stance, code-mixing, Romanisation) cause it.
4. **Minority-class behaviour.** In every dataset we measured or read, "against" is rare (4.6% of SemEval climate, 14.3%
   of our English training set; Dhoroni "negative" 6%) and is the class models recover worst, yet macro-F1 weights it equally.

## 8. Proposed methodology

Staged so that every stage is explainable and measurable:

1. **Normalise** (NFC, strip URLs/mentions/timestamps/zero-width characters, collapse 3+ repeats per code point, keep emoji)
   and identify script (Devanagari / Bengali / Latin / mixed) from Unicode ranges.
2. **Floor and baselines.** Majority-class floor; TF-IDF word 1-2 + char_wb 2-5 with LogReg / LinearSVC, class-weighted.
   *Done in Milestone 1.*
3. **Target-aware multilingual encoder.** Fine-tune MuRIL (XLM-R and IndicBERT as comparisons) on the pair
   `claim [SEP] comment`, class-weighted loss, macro-F1 for model selection, fixed seed. *Script written, not yet run.*
4. **Cross-lingual strategy.** Compare zero-shot transfer from English, translate-train (IndicTrans2/NLLB), and English +
   translated data; add English external data (GWSD, SemEval) following the CASE@EACL finding. *Planned.*
5. **Evaluation.** Macro-F1 primary; accuracy, per-class precision/recall and confusion matrix supporting; per-phenomenon and
   per-script breakdown; a zero-shot LLM baseline given the 2309.13734 caveat. The pilot corpus is for development only;
   no number from it is reported as an official result.

---

## References

1. Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., Cherry, C. (2016). SemEval-2016 Task 6: Detecting Stance in Tweets. *SemEval-2016*. [recall; data read via TweetEval]
2. Luo, Y., Card, D., Jurafsky, D. (2020). Detecting Stance in Media on Global Warming. *Findings of EMNLP 2020*. [recall]
3. Shiwakoti, S. et al. (2024). Analyzing the Dynamics of Climate Change Discourse on Twitter: A New Annotated Corpus and Multi-Aspect Classification (ClimaConvo). *LREC-COLING 2024*. Repository github.com/shucoll/ClimaConvo. [README and data file read; title from memory, verify title and author list]
4. Thapa, S. et al. (2024). Stance and Hate Event Detection in Tweets Related to Climate Activism: Shared Task at CASE 2024. *CASE@EACL 2024*. [recall; findings as in task brief]
5. Wang, J., Zuo, L., Peng, S., Plank, B. (2024). MultiClimate: Multimodal Stance Detection on Climate Change Videos. arXiv:2409.18346 (NLP4PI workshop, EMNLP 2024). [read]
6. Wasi, A. T., Faisal, W., Ahmad, T., Rahman, A., Islam, M. R. (2024). Dhoroni: Exploring Bengali Climate Change and Environmental Views with a Multi-Perspective News Dataset and Natural Language Processing. arXiv:2410.17225. [read]
7. Swami, S., Khandelwal, A., Singh, V., Akhtar, S. S., Shrivastava, M. (2018). An English-Hindi Code-Mixed Corpus: Stance Annotation and Baseline System. arXiv:1805.11868. [read]
8. Khanuja, S. et al. (2021). MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730. [recall]
9. Kakwani, D. et al. (2020). IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages. *Findings of EMNLP 2020*. [recall]
10. Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *ACL 2020*. [recall]
11. Aggarwal, D. et al. (2022). IndicXNLI: Evaluating Multilingual Inference for Indian Languages. arXiv:2204.08776 (EMNLP 2022). [recall; findings as in task brief]
12. Pangtey, L. et al. (2025). Large Language Models Meet Stance Detection: A Survey. arXiv:2505.08464. [recall]
13. Cross-target stance detection survey. arXiv:2409.13594. [recall; verify title and authors]
14. Cruickshank, I. J., Ng, L. H. X. (2023). Prompting and Fine-Tuning Open-Sourced Large Language Models for Stance Classification. arXiv:2309.13734. [recall; verify]
15. Küçük, D., Can, F. (2020). Stance Detection: A Survey. *ACM Computing Surveys* 53(1). [recall]
16. StanceEval 2024 / 2026 shared tasks on Arabic stance detection (stanceeval.github.io). [recall]
17. Squad4 (2026). Translate-Train Climate-Change Stance Detection in Hindi and Bengali, AISoMe 2026 at FIRE 2026. Repository github.com/Debajyoti212/Aisome2026. [read: reported scores only; competing team, no code or data reused]
18. AISoMe 2026 track at FIRE 2026, Kolkata (task description: Hindi/Bengali YouTube comments, Favour/Against/None, English training data). [read via search summary]
