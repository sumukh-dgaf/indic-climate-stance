# Eklavya Sharma: Bengali review and Hindi-vs-Bengali analysis

You own the Bengali side of the project. In the evaluation you explain how good our Bengali data and labels are, and
why the models do worse on Bengali than on Hindi. Everything here works in a browser and Google Sheets: no installation.

## Tasks

1. **Review the synthetic test set (Bengali rows).** Wait until Neya has uploaded her Hindi rows (two people must not
   edit the file at the same time). Then download `data/review/synthetic_test_review.csv` from GitHub and open it in
   Google Sheets (not Excel, which garbles Bengali). For every row with `language = bn` (300 rows):
   * `reviewer`: your name;
   * `reviewer_label`: `Favour`, `Against` or `None`, i.e. the stance toward *"Climate change and global warming is a
     serious concern."* as you read it. The intended label is deliberately hidden: do not look it up;
   * `comment_is_natural`: `yes` if a real person could plausibly have written it, else `no`.

   Do not change Neya's `hi` rows. Download as CSV with the same name and upload it to `data/review/` on GitHub
   (Add file > Upload files > Commit changes).
2. **Check the Bengali translations.** Sumukh sends you `translation_review_bn.csv` privately. Compare `english` with
   `translation` and fill `reviewer`, `label_survives` (`yes` if the `stance` label is still correct for the Bengali
   translation, `no` if translation changed or lost it) and `notes` (why, for every `no`). Send the file back to Sumukh.
   It contains GWStance/SemEval text with no redistribution licence: **do not upload it to GitHub**.
3. **Write the Hindi-vs-Bengali analysis** in your own words in `docs/bengali_analysis.md` (150 to 250 words; on GitHub:
   `docs/` > Add file > Create new file). Cover:
   * the gap between languages: section "Macro-F1 by language" in `outputs/round2_results.md` (Bengali is lower than
     Hindi on every test set);
   * the Bengali rows in `outputs/error_examples.csv` (most of the 15 misclassified examples are Bengali): pick two or
     three and explain what went wrong;
   * what your review found about the Bengali comments: how natural they were, and how often you disagreed with the
     intended label;
   * one or two possible reasons for the gap (for example less Bengali pre-training data, translation quality, script
     and spelling variation).

   Use only numbers from `outputs/round2_results.md` and your own review. Aditya owns editing
   `outputs/error_examples.csv`; you only read it.

## Be ready to explain
* How the blind review works and why the intended label is hidden from the reviewer.
* How the code tells Bengali from Hindi text (Unicode script ranges: Bengali U+0980-U+09FF, Devanagari U+0900-U+097F).
* Why Bengali scores are lower than Hindi, with numbers from the results.

## Your account

Accept Sumukh's GitHub invite from your email and make sure that email is on your GitHub account (GitHub > Settings >
Emails), so your uploads count as your contributions.
