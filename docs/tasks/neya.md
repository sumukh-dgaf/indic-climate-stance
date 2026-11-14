# Neya Tripathi: dataset generation and translation

You own the data. In the evaluation you explain where every row comes from, why it is labelled the way it is, and how
good the labels are. Label honestly everywhere: human-labelled English, machine-translated, or synthetic
(LLM-generated). The synthetic comments are not real YouTube comments.

## Tasks

1. **Review the synthetic test set (Hindi rows).** Open `data/review/synthetic_test_review.csv` in Excel or LibreOffice
   (save as CSV, UTF-8). For every row with `language = hi`:
   * `reviewer`: your name;
   * `reviewer_label`: `Favour`, `Against` or `None`, i.e. the stance toward *"Climate change and global warming is a
     serious concern."* as you read it. The intended label is deliberately hidden: do not look it up;
   * `comment_is_natural`: `yes` if a real person could plausibly have written it, else `no`.

   The `bn` rows are labelled by whichever member reads Bengali, in the same file.
2. **Check the translations.** In `data/review/translation_review_hi.csv` (Bengali reader: `_bn.csv`), compare `english`
   with `translation` and fill `label_survives` = `yes` if the `stance` label is still correct for the translation,
   `no` if translation changed or lost the stance; use `notes` for why. These two files contain GWStance/SemEval text
   with no redistribution licence: **do not commit them**.
3. **Apply the reviews** (after both languages are done):
   ```bash
   python src/apply_review.py                 # replaces disagreeing labels, sets human_verified=yes -> data/review/review_stats.json
   python src/apply_review.py --translation   # counts only -> data/review/translation_review_stats.json
   python src/evaluate.py                     # results now state whether the synthetic test labels are verified
   git add data/review/synthetic_test_review.csv data/review/review_stats.json data/review/translation_review_stats.json data/synthetic/indic_synthetic_v1.csv outputs/
   git commit -m "Apply human review of synthetic test labels and translation check"
   git push
   ```
4. **Write the dataset statistics paragraph** in your own words in `docs/dataset_statistics.md` (120 to 180 words):
   what each data family is, its size and class distribution (`outputs/round2_results.json` > `dataset`), the judge
   agreement rate (`data/synthetic/generation_stats.json`) and what your review found. Use only numbers that appear in
   those files. Commit and push it.

## Be ready to explain
* Why the judge model differs from the generator model, and what the agreement rate per phenomenon tells us.
* Why translation happens after the split (a test sentence's translation must not end up in training).
* Why configuration C's score on the synthetic test set is optimistic.

## One-time setup (your own account)

Sumukh adds you as a collaborator (GitHub > repository > Settings > Collaborators). Then:

```bash
git clone https://github.com/sumukh-dgaf/indic-climate-stance.git
cd indic-climate-stance
git config user.name  "<your full name>"
git config user.email "<your Bennett email>"
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Ask Sumukh privately for the files that are not on GitHub (licence or size): `data/review/translation_review_{hi,bn}.csv`.
