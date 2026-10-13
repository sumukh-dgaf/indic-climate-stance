# data/ : what is committed, what is not, and how to rebuild it

## Committed
* `pilot_indic_stance.csv`: 150 **synthetic, team-authored** Hindi/Bengali comments (development only; not scraped, not the
  official data).
* `DATASET_INVENTORY.md`: every source examined, its status, and the target/language/domain gap table.

## Not committed (git-ignored), and why

| File | Why not committed |
|---|---|
| `english_stance_merged.csv`, `english_stance_highconf.csv`, `gwstance_ambiguous_ties.csv` | They contain GWSD text. **The GWStance repository (github.com/yiweiluo/GWStance) has no LICENSE file** (every common license filename returns 404 and its README states no terms), so no redistribution right can be assumed. The files also contain SemEval-2016 tweets. |
| `external/`, `raw/` | Third-party downloads (Dhoroni, MultiClimate, Swami et al., TweetEval, GWSD.tsv); re-fetchable. |

Path taken: **keep ignored and make the files exactly reproducible from source**, rather than redistribute data we have no
licence to redistribute.

## Rebuild the two English files (needs internet)

```bash
pip install -r requirements.txt
python src/build_english_merged.py            # fetches GWSD.tsv + SemEval climate (TweetEval); writes the 3 CSVs
python src/build_english_merged.py --check    # on a machine that already has the files: compares field by field
```

The script refuses to overwrite existing files unless `--force` is given. Its `--check` mode was run against the team's
supplied files and reported **0 differences in every field** (same 2,610 / 1,686 rows, same ids, labels, splits and
agreement values).

### The exact rules (implemented in `src/build_english_merged.py`)

1. **GWSD** (`GWSD.tsv`, 2,300 rows, columns `worker_0` ... `worker_7` each `agrees` / `disagrees` / `neutral`):
   * label each worker vote `agrees -> Favour`, `disagrees -> Against`, `neutral -> None`;
   * **majority-vote rule:** `stance` = the label with the **unique highest count** among the 8 votes;
     `agreement` = that count / 8 (so it is 0.375 ... 1.0);
   * **tie rule:** if two labels share the highest count (4-4, or 3-3-2), `stance` is left **blank**. On the raw file this
     is **215 of 2,300 rows (9.3%)**;
   * `split` = `test` if `in_held_out_test` is True (200 rows) else `train`; ids are `gwstance_<guid>`; comments are
     whitespace-stripped.
2. **SemEval-2016 climate** via `cardiffnlp/tweet_eval`, config `stance_climate` (train 355 / validation 40 / test 169):
   published gold label (`favor -> Favour`, `against -> Against`, `none -> None`), `agreement = gold`, the `#SemST` tag
   removed, ids `climate_<split>_<i>`.
3. Concatenate GWSD then SemEval, then **drop rows with duplicate comment text (first kept)**. This removes 253 GWSD rows
   including 13 ties, leaving 2,047 GWSD rows (202 ties) and 563 SemEval rows = **2,610 rows**.
4. `english_stance_highconf.csv` = GWSD rows with `agreement >= 0.75` plus all SemEval rows = **1,686 rows**.
5. **Loading:** `src/data.py::load_english` drops the 202 blank-stance rows (explicitly, with a printed count, and writes
   them to `gwstance_ambiguous_ties.csv`), leaving **2,408 labelled rows**. Always read labelled files with
   `keep_default_na=False`: pandas otherwise converts the label string `"None"` to NaN and silently deletes a class.

### Rebuild the external sources

```bash
python src/fetch_external.py     # -> data/external/*.csv (SemEval climate, Dhoroni, MultiClimate, Swami et al.)
```
