# Aaditya Goel: evaluation and error analysis

You own the numbers. In the evaluation you explain what macro-F1 is, why we report confidence intervals, and what the
model gets wrong.

## Tasks

1. **Run the evaluation** and read every table:
   ```bash
   python src/evaluate.py      # reads outputs/preds/ -> outputs/round2_results.md/.json, fig_r2_*.png, error_examples.csv
   ```
2. **Study the errors.** Open `outputs/error_examples.csv` (15 misclassified rows of the best model from the synthetic
   test set and the pilot, grouped by phenomenon). For each row fill `team_notes` with a short reason: sarcasm missed,
   negation, no climate keyword, Romanised spelling, the label itself is debatable, and so on.
3. **Write the error-analysis section** in your own words in `docs/error_analysis.md` (200 to 300 words): which
   phenomena fail most (table "Accuracy by phenomenon" in `outputs/round2_results.md`), which class is confused with
   which (confusion matrices, `outputs/fig_r2_confusion_best.png`), and two or three examples with your explanation.
   Use only numbers from `outputs/round2_results.json`.
4. Commit and push both:
   ```bash
   git add outputs/error_examples.csv docs/error_analysis.md
   git commit -m "Error analysis of the best Round 2 model"
   git push
   ```

## Be ready to explain
* Macro-F1 vs accuracy, and why the majority-class floor is shown next to every score.
* What a 95% bootstrap confidence interval is (1,000 resamples of the test rows) and why overlapping intervals mean
  "no clear difference".
* Why the best model is chosen on the translated and pilot sets, not on the synthetic test set.

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

`evaluate.py` needs only `outputs/preds/` (committed). Its dataset table also reads `data/splits/` and `data/translated/`
(not on GitHub for licence reasons): ask Sumukh for them privately, or run it on Sumukh's machine.
