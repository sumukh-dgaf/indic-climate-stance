"""Pack what Colab needs into colab_bundle.zip (git-ignored: it contains the GWStance-derived English splits).

Run: python src/make_colab_bundle.py      then upload colab_bundle.zip to Google Drive folder MyDrive/indic_stance/
"""
import glob
import os
import sys
import zipfile

REQUIRED = ["src/*.py", "requirements*.txt", "data/splits/english_*.csv", "data/pilot_indic_stance.csv"]
SYNTHETIC = "data/synthetic/indic_synthetic_v1.csv"
OUT = "colab_bundle.zip"


def files():
    found, missing = [], []
    for pattern in REQUIRED:
        hits = sorted(glob.glob(pattern))
        (found.extend(hits) if hits else missing.append(pattern))
    return found, missing


def main():
    found, missing = files()
    if missing:
        sys.exit(f"missing {missing}: run python src/make_splits.py first")
    if os.path.exists(SYNTHETIC):
        found.append(SYNTHETIC)
    else:
        print(f"!! {SYNTHETIC} not found: C runs will FAIL on Colab (A and B still run; re-run after generating it)")
    assert len(glob.glob("data/splits/english_*.csv")) == 3
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in found:
            z.write(f, f.replace(os.sep, "/"))
    names = zipfile.ZipFile(OUT).namelist()
    assert "src/run_all.py" in names and "data/splits/english_test.csv" in names
    print(f"wrote {OUT}: {len(names)} files, {os.path.getsize(OUT) / 1e6:.1f} MB")
    for n in names:
        if not n.startswith("src/"):
            print("  ", n)


if __name__ == "__main__":
    main()
