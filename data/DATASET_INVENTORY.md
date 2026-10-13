# Dataset inventory

One row per source examined for Milestone 1. "Rows obtained" are rows that exist as text on this machine, counted by code
(`src/fetch_external.py`, `src/data.py`), not figures copied from papers. Where the two differ it is stated.

**Target of our task:** "Climate change and global warming is a serious concern." Labels Favour / Against / None.
**Languages:** Hindi and Bengali comments (YouTube / social media).

## 1. Sources

| # | Source | Citation | Language | Domain | Stance target | Rows obtained | Label mapping applied | Access route | Role |
|---|---|---|---|---|---|---|---|---|---|
| 1 | SemEval-2016 Task 6, climate target (via `cardiffnlp/tweet_eval`, config `stance_climate`) | Mohammad et al. 2016 | English | Tweets | "Climate Change is a Real Concern" | **564** (355 train / 40 val / 169 test); Favour 335, None 203, **Against 26** | favor->Favour, against->Against, none->None | Hugging Face `datasets` | **Training / transfer data** |
| 1b | SemEval-2016 Task 6 full mirror `krishnagarg09/SemEval2016Task6` | Mohammad et al. 2016 | English | Tweets | 4 other targets | **0 climate rows** (Hillary, Feminist Movement, Abortion, Atheism only) | - | Hugging Face | Unusable for us |
| 2 | GWStance / GWSD, merged with SemEval climate by the team (`data/english_stance_merged.csv`) | Luo, Card & Jurafsky 2020 (GWSD); Mohammad et al. 2016 | English | **News sentences (2,047) + tweets (563)** | "global warming is a serious concern" (GWSD, verbatim); "Climate Change is a Real Concern" (SemEval) | 2,610 supplied; **202 annotator ties dropped -> 2,408 labelled** (Favour 1,034 / Against 345 / None 1,029) | already in Favour/Against/None; majority vote of 8 annotators | Supplied by team member | **Primary training data** |
| 2b | Same, high-confidence subset (`data/english_stance_highconf.csv`) | as above | English | as above | as above | 1,686 (Favour 792 / Against 222 / None 672); GWSD rows with agreement >= 0.75 | as above | Supplied by team member | Training (cleaner, smaller) |
| 3 | ClimaConvo | Shiwakoti et al. 2024 (LREC-COLING) | English | Tweets | Climate change (stance: support / denial / neutral) | 15,309 **IDs + labels only; no text** (longest cell 22 characters) | not applicable | GitHub `shucoll/ClimaConvo`; text needs the paid X API | **Methodology evidence only** |
| 4 | Dhoroni | Wasi et al. 2024 (arXiv 2410.17225) | **Bengali** | **News articles (mean 2,570 characters)** | "supports climate change as a real and pressing issue" | 2,300 articles -> **2,193 mapped** (Favour 1,586 / Against 137 / None 470); 107 "Problematic" dropped | Positive->Favour, Negative->Against, Neutral->None | Hugging Face `ciol-research/Dhoroni` (MIT) | **Transfer / domain-adaptation evidence; not comments** |
| 5 | MultiClimate | Wang et al. 2024 (arXiv 2409.18346) | English | YouTube **video transcript sentences**, not comments | climate change (attitude of the frame-transcript pair) | 4,209 raw -> **4,144 non-empty** (support 1,847 / oppose 1,176 / neutral 1,121) | support->Favour, oppose->Against (**low confidence**), neutral->None | GitHub `werywjw/MultiClimate` (per-video CSVs) | **Methodology evidence only** |
| 6 | English-Hindi code-mixed stance corpus | Swami et al. 2018 (arXiv 1805.11868) | Romanised Hindi + English | Tweets | **Demonetisation, NOT climate** | 3,545 (FAVOR 964 / AGAINST 647 / NONE 1,934), matches the paper | FAVOR->Favour etc. (**same names, different meaning**) | GitHub `sahilswami96/StanceDetection_CodeMixed` (GPL-3.0) | **Kept separate. Code-mixed methodology evidence; possible Milestone 2 domain-adaptive pretraining. Never in the climate pool** |
| 7 | Hindi Climate-FEVER (`AIhnIndicRag/Hindi_Climate_Fever`) | - | Hindi | Encyclopedic | - | 4,552 retrieval qrels (`query-id, corpus-id, score`) | none: no stance labels | Hugging Face | **Not usable for stance.** At most Hindi climate-domain text for future pretraining |
| 8 | `mlburnham/global_warming_stance_entailment` | - | English | Mixed sentences, GPT-4o-validated | "climate change is a serious concern" / "...is not a concern" (NLI format) | 3,607, opened, **not used** | - | Hugging Face | Noted only |
| 9 | **Pilot corpus (ours)** `data/pilot_indic_stance.csv` | team-authored | Hindi, Bengali (incl. code-mixed and Romanised) | Synthetic YouTube-comment register | our claim | 150 (50/50/50; 75 Hindi + 75 Bengali) | written in Favour/Against/None | written by the team | **Development set only. Synthetic: not scraped, not the official data** |
| 10 | AISoMe 2026 official data | FIRE 2026 shared task | English train; Hindi + Bengali test | YouTube comments | our claim | **0: not in hand** | - | emailed to an inaccessible address | The data this project is ultimately evaluated on |

## 2. Third-party results used only as related work

The AISoMe 2026 participant repository `Debajyoti212/Aisome2026` (team "Squad4") was found while searching. It is a
**competing team's** work. We use only the scores it reports, cited by name in the literature review and the SOTA table.
We did not copy, adapt or consult its code, data pipeline or methodology, and deliberately limited our use to the
published numbers.

## 3. No existing resource matches target AND language AND domain

Axes: does the stance **target** match our claim; is the **language** Hindi/Bengali; is the **domain** social-media comments.

| Resource | Target = our claim | Hindi / Bengali | Comment-style social media | Text obtainable | All three |
|---|---|---|---|---|---|
| SemEval-2016 climate | yes | no (English) | yes (tweets) | yes | **no** |
| GWSD (GWStance) | yes (verbatim) | no (English) | no (news sentences) | yes | **no** |
| ClimaConvo | close (climate stance) | no (English) | yes (tweets) | **no (IDs only)** | **no** |
| MultiClimate | close (climate attitude) | no (English) | no (video transcripts) | yes | **no** |
| Dhoroni | close ("real and pressing issue") | **yes (Bengali)** | no (news articles) | yes | **no** |
| Swami et al. 2018 | **no (demonetisation)** | partly (Romanised Hindi) | yes (tweets) | yes | **no** |
| Hindi Climate-FEVER | no (retrieval, no stance) | yes (Hindi) | no | yes | **no** |
| Our pilot corpus | yes | yes | register imitated, 150 synthetic rows | yes | **not real data** |
| AISoMe official test | yes | yes | yes | **hidden / not in hand** | yes, but unavailable for training |

The same information as a 2 x 2 (target x language), with the failing axis noted in brackets:

|  | **Target = our claim** | **Target differs** |
|---|---|---|
| **English** | SemEval-2016 climate (tweets), GWSD (news) -- *[language]* | ClimaConvo (no text), MultiClimate (video transcripts) -- *[language, text/domain]* |
| **Hindi / Bengali** | Dhoroni (Bengali news, near-match target) -- *[domain]*; our synthetic pilot -- *[not real]* | Swami (Romanised Hindi tweets, demonetisation) -- *[target]*; Hindi Climate-FEVER -- *[no stance labels]* |

**Conclusion (from evidence above):** among the sources we could find and open, none provides Hindi or Bengali
social-media comments labelled for stance on our claim. The English side supplies target-correct labels; the Indic side
supplies language or domain but not the target. This is the research gap, and it is why the task is cross-lingual
transfer.
