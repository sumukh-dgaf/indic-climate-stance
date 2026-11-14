# Kushika Mishra: report, deck and demo

You own what the evaluators see: the two-page report, the slides and the live demo.

## Tasks

1. **Edit the progress report.** `python src/make_report.py` builds `docs/Round2_Progress_Report.docx` (exactly two A4
   pages; every number is read from `outputs/round2_results.json`). Read it critically, improve the wording inside
   `src/make_report.py` (not in the .docx, so the numbers stay generated), rebuild, and check it is still two pages.
   Never type a number by hand.
2. **Edit the deck.** `python src/make_deck.py` builds `outputs/Round2_Deck.pptx` from the same JSON. Improve titles and
   speaker notes the same way.
3. **Test the demo on new comments.** Write 20 new comments of your own (Hindi, Bengali if you can, code-mixed,
   Romanised, sarcastic, off-topic), run each one, and record the result in `docs/demo_test_log.md`:
   ```bash
   python src/demo.py --examples                 # the fixed examples
   python src/demo.py "your comment here"        # one comment
   python src/demo.py                            # interactive prompt
   ```
   Table columns: comment | type | your label | model label | probabilities | correct? Then write three sentences on
   what it gets wrong.
4. Commit and push:
   ```bash
   git add src/make_report.py docs/Round2_Progress_Report.docx src/make_deck.py outputs/Round2_Deck.pptx docs/demo_test_log.md
   git commit -m "Edit Round 2 report and deck; demo test log"
   git push
   ```

## Be ready to explain
* What the demo prints (label + three probabilities) and what happens if the transformer folder is missing (TF-IDF
  fallback, announced on screen).
* The five report headings and where each number in them comes from.

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

For the demo, ask Sumukh for the trained model folder `models/indicbert_C_full/` (about 1 GB, not on GitHub) and put
it in `models/`. Without it the demo still runs, on the TF-IDF fallback.
