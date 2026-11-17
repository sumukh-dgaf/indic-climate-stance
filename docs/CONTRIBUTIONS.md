# Individual contributions (Round 2)

Each member owns one area: they can explain it in the evaluation, run its code live, and commit the work listed in
their task file from their **own** GitHub account. Who committed what is visible in the git history
(`git log --format='%an  %s'`); nobody commits under another member's name.

| Member | Enrolment | Area owned | Task file |
|---|---|---|---|
| Sumukh Gupta | S24CSEU0570 | Models and training: data splits, the three training configurations, IndicBERT v2 / MuRIL / TF-IDF runs, the run schedule | [docs/tasks/sumukh.md](tasks/sumukh.md) |
| Neya Tripathi | S24CSEU0609 | Dataset generation and translation: the synthetic Gemini dataset, the machine-translated data, the human review of both, dataset statistics | [docs/tasks/neya.md](tasks/neya.md) |
| Aditya Goel | S24CSEU0614 | Evaluation and error analysis: metrics, confidence intervals, the three comparisons, the misclassified-example study | [docs/tasks/aditya.md](tasks/aditya.md) |
| Kushika Mishra | S24CSEU0652 | Report, deck and demo: the two-page progress report, the slides, testing the live demo on new comments | [docs/tasks/kushika.md](tasks/kushika.md) |
| Eklavya Sharma | S24CSEU0798 | Bengali review and Hindi-vs-Bengali analysis: the blind review of the 300 Bengali synthetic test rows, the Bengali translation check, and the comparison of model performance on Hindi vs Bengali | [docs/tasks/eklavya.md](tasks/eklavya.md) |

## What each area covers in the repository

* **Models and training (Sumukh):** `src/make_splits.py`, `src/train_transformer.py`, `src/baseline.py --config`,
  `src/predict.py`, `src/run_all.py`, `notebooks/round2_colab.ipynb`.
* **Dataset generation and translation (Neya):** `src/generate_synthetic.py`, `src/translate.py`, `src/apply_review.py`,
  `data/synthetic/`, `data/review/`.
* **Evaluation and error analysis (Aditya):** `src/evaluate.py`, `outputs/round2_results.md/.json`,
  `outputs/error_examples.csv`, `docs/error_analysis.md`.
* **Report, deck and demo (Kushika):** `src/make_report.py`, `docs/Round2_Progress_Report.docx`, `src/make_deck.py`,
  `outputs/Round2_Deck.pptx`, `src/demo.py`, `docs/demo_test_log.md`.
* **Bengali review and Hindi-vs-Bengali analysis (Eklavya):** the `bn` rows of `data/review/synthetic_test_review.csv`,
  the Bengali translation check, `docs/bengali_analysis.md`.

The Round 2 pipeline code was first committed from Sumukh's account (see git history). The human review, the written
analysis, the demo testing and the report and deck editing are the members' own work, committed by them.

## Bengali review

The Bengali rows of `data/review/synthetic_test_review.csv` and `translation_review_bn.csv` are reviewed by Eklavya Sharma,
after Neya has uploaded the Hindi rows (two people must not edit the review file at the same time). Every reviewer writes
their name in the `reviewer` column of each row they label.
