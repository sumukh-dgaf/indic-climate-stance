# Sumukh Gupta: models and training

You own the models. In the evaluation you run training live and explain the method.

## Tasks

1. Reproduce from scratch and be able to run any step live:
   ```bash
   python src/make_splits.py
   python src/train_transformer.py --smoke                               # whole path on CPU, about 75 s
   python src/train_transformer.py --model indicbert --config A_english  # real run on the RTX 5050
   python -u src/run_all.py                                              # whole schedule; finished runs are skipped
   ```
2. Add the three teammates as GitHub collaborators and share privately the git-ignored files they need:
   `models/indicbert_C_full/`, `data/translated/`, `data/splits/`, `data/review/translation_review_*.csv`.
3. After teammates push the review, re-run `python src/evaluate.py`, `python src/make_report.py` and
   `python src/make_deck.py`, and commit the regenerated outputs.

## Be ready to explain
* The pair input (English claim, comment) and why the claim stays English for every language.
* Class-weighted cross-entropy, 10% warmup, mixed precision, model selection on the configuration's own validation data.
* What configurations A, B and C each test, and why A and B are the fairer measure of cross-lingual transfer.
