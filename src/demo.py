"""Live demo: classify Hindi/Bengali comments toward "Climate change and global warming is a serious concern."

  python src/demo.py "ये सब पश्चिमी देशों का प्रोपेगेंडा है"     one comment: label + three probabilities
  python src/demo.py --examples                               ten fixed examples (Hindi, Bengali, code-mixed, Romanised, ...)
  python src/demo.py                                          interactive prompt (empty line or Ctrl+C to quit)
Loads models/indicbert_C_full/. If that is missing or broken it falls back to TF-IDF + LogReg AND SAYS SO, so the demo
never fails outright. Self-check: python src/demo.py --check
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import LABELS  # noqa: E402
from preprocess import clean  # noqa: E402

MODEL_DIR = "models/indicbert_C_full"
TFIDF = ["models/tfidf_C_full.joblib", "models/tfidf_B_translated.joblib", "models/tfidf_A_english.joblib"]
# Demo inputs written for the live demo (not part of any dataset). "written as" = the stance the author intended.
EXAMPLES = [
    ("हर साल गर्मी बढ़ती जा रही है, अब तो सच में डर लगने लगा है।", "Hindi", "Favour"),
    ("ये सब पश्चिमी देशों का प्रोपेगेंडा है, गर्मी तो हमेशा से पड़ती आई है।", "Hindi", "Against"),
    ("হিমালয়ের হিমবাহ গলে যাচ্ছে, এখনই কিছু না করলে আমাদের নদীগুলো শুকিয়ে যাবে।", "Bengali", "Favour"),
    ("এবার এত ঠান্ডা পড়েছে, আর ওরা বলছে পৃথিবী গরম হচ্ছে! হাস্যকর।", "Bengali", "Against"),
    ("Bhai ये heatwave normal नहीं है, climate change real है और हम ignore कर रहे हैं", "code-mixed Hindi", "Favour"),
    ("global warming kuch nahi hai yaar, scientists ko bas funding chahiye", "Romanised Hindi", "Against"),
    ("ei bonna ar cyclone dekhe bujhte parchi, ekhoni sabdhan na hole sob sesh", "Romanised Bengali", "Favour"),
    ("हाँ हाँ, दिसंबर में रज़ाई ओढ़ के बैठा हूँ, बहुत ग्लोबल वार्मिंग हो रही है 😂", "sarcastic Hindi", "Against"),
    ("Nice video bhai, background music ka naam kya hai?", "off-topic, Romanised", "None"),
    ("৩:৪৫ এ যে জায়গাটা দেখাল ওটা কোথায়? যেতে চাই", "off-topic Bengali", "None"),
    ("क्या कोई बता सकता है कि अल नीनो क्या होता है?", "question, Hindi", "None"),
]


def tfidf_classifier():
    import joblib
    for p in TFIDF:
        if os.path.exists(p):
            m = joblib.load(p)  # pickle: only files our own baseline.py wrote locally (models/ is git-ignored, never downloaded)
            return f"FALLBACK: TF-IDF + LogReg ({p})", m
    print("!! no saved TF-IDF model either: training TF-IDF + LogReg on the English data now (a few seconds)")
    from baseline import model
    from data import config_data, load_english
    try:
        df = config_data("A_english", "train")
    except FileNotFoundError:
        df = load_english()
    return "FALLBACK: TF-IDF + LogReg trained just now on English only", model("word+char", "LogReg").fit(df.comment.map(clean), df.stance)


def load_classifier():
    """Returns (description, function texts -> (n, 3) probabilities in LABELS order)."""
    if os.path.isdir(MODEL_DIR):
        try:
            from predict import load_model, predict_probs
            tok, net, info = load_model(MODEL_DIR)
            return f"IndicBERT v2 fine-tuned on configuration C ({MODEL_DIR})", \
                lambda texts: predict_probs(net, tok, texts, info["max_len"], "cpu")
        except Exception as e:  # broken download, missing torch...: degrade, never crash on stage
            print(f"!! could not load {MODEL_DIR} ({type(e).__name__}: {str(e)[:120]})")
    else:
        print(f"!! {MODEL_DIR} not found (unzip colab_results.zip in the repo root)")
    desc, m = tfidf_classifier()
    order = [list(m.classes_).index(lab) for lab in LABELS]
    return desc, lambda texts: m.predict_proba([clean(t) for t in texts])[:, order]


def show(text, probs, note=""):
    i = int(probs.argmax())
    print(f"\n{text}\n  -> {LABELS[i]:7s}  " + "  ".join(f"{lab} {p:.2f}" for lab, p in zip(LABELS, probs)) + note)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Devanagari/Bengali on a Windows console or pipe
    args = sys.argv[1:]
    desc, clf = load_classifier()
    print(f"Model: {desc}")
    if desc.startswith("FALLBACK"):
        print("   (the transformer is not loaded: these are the BASELINE's predictions)")
    if args == ["--examples"]:
        probs = clf([t for t, _, _ in EXAMPLES])
        hits = 0
        for (t, kind, written), p in zip(EXAMPLES, probs):
            hits += LABELS[int(p.argmax())] == written
            show(t, p, f"   [{kind}; written as {written}]")
        print(f"\n{hits} of {len(EXAMPLES)} match the stance the example was written with")
    elif args:
        text = " ".join(args)
        show(text, clf([text])[0])
    else:
        print("Type a comment (empty line to quit).")
        while True:
            try:
                text = input("\ncomment> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            show(text, clf([text])[0])


def selfcheck():
    assert len(EXAMPLES) >= 10 and {w for _, _, w in EXAMPLES} == set(LABELS)
    desc, clf = load_classifier()
    p = clf(["global warming kuch nahi hai yaar", "nice video"])
    assert p.shape == (2, 3) and abs(p.sum(1) - 1).max() < 1e-4
    print(f"demo.py self-check passed using: {desc}")


if __name__ == "__main__":
    selfcheck() if sys.argv[1:] == ["--check"] else main()
