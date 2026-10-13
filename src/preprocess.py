"""Text cleaning and script-based language ID. Self-check: python src/preprocess.py"""
import re
import unicodedata

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")  # \d is Unicode-aware, so Bengali/Devanagari digits match too
ZERO_WIDTH = dict.fromkeys([0x200B, 0x200C, 0x200D])  # ZWSP, ZWNJ, ZWJ -> deleted by str.translate
# Collapse 3+ identical NON-digit codepoints to 2 ("1000000" must survive). This works on CODEPOINTS, not glyphs:
# in Devanagari a vowel sign such as ी is its own codepoint, so नहीीीीी (5x ी) becomes नहीी, not नही.
REPEAT = re.compile(r"(\D)\1{2,}")


# Word tokens. Python's \w (and sklearn's default token_pattern) does NOT match Indic vowel signs (category Mc/Mn),
# so "जलवायु" would be cut into ['जलव', 'य']. We add the Devanagari/Bengali blocks explicitly, minus the danda U+0964/5.
TOKEN = re.compile(r"[\w\u0900-\u0963\u0966-\u09FF]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def clean(text):
    text = unicodedata.normalize("NFC", text)  # FIRST: nukta letters (e.g. क़) have a precomposed and a decomposed encoding
    text = text.translate(ZERO_WIDTH)  # ponytail: also splits emoji ZWJ sequences and some Bengali conjuncts; fine for classification
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = TIMESTAMP.sub(" ", text)
    text = text.replace("#SemST", " ")  # SemEval-2016 collection artefact appended to every tweet, carries no stance
    text = REPEAT.sub(r"\1\1", text)  # emoji are kept (they carry stance); 😂😂😂 -> 😂😂
    return re.sub(r"\s+", " ", text).strip()


def script_id(text, dominant=0.8):
    """hi / bn / latin from Unicode ranges only; 'mixed' if no script holds >= 80% of letters; 'other' if no letters."""
    counts = {"hi": 0, "bn": 0, "latin": 0}
    for ch in text:
        cp = ord(ch)
        # letters (L*) and vowel signs/marks (M*) count; digits, danda, emoji do not
        if unicodedata.category(ch)[0] not in "LM":
            continue
        if 0x0900 <= cp <= 0x097F:
            counts["hi"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            counts["bn"] += 1
        elif cp < 0x250 and ch.isalpha():
            counts["latin"] += 1
    total = sum(counts.values())
    if total == 0:
        return "other"
    best = max(counts, key=counts.get)
    # ponytail: Romanised Hindi/Bengali and English are both 'latin'; telling them apart needs a lexical model
    return best if counts[best] / total >= dominant else "mixed"


if __name__ == "__main__":
    # gotcha 3: NFC first. U+0958 (क़ precomposed) vs U+0915 U+093C (क + nukta) look identical but differ as strings
    a, b = "क़", "क़"
    assert a != b and clean(a) == clean(b)
    # gotcha 2: repetition collapse is per codepoint
    assert clean("नहीीीीी") == "नहीी", clean("नहीीीीी")
    assert clean("soooo goooood") == "soo good"
    assert clean("1000000 log") == "1000000 log"  # digits are not collapsed
    # noise stripping, emoji kept
    assert clean("see https://t.co/x @user 12:45 ok\u200b") == "see ok"
    assert clean("১২:৪৫ এ যা বলল") == "এ যা বলল"  # Bengali-digit timestamp
    assert clean("Global warming #SemST") == "Global warming"
    assert clean("हाहा 😂😂😂😂") == "हाहा 😂😂"
    # Indic-specific finding: plain \w+ fragments Hindi words at vowel signs; our TOKEN pattern does not
    assert len(re.findall(r"\w+", "जलवायु")) > 1
    assert tokenize("जलवायु परिवर्तन गंभीर है।") == ["जलवायु", "परिवर्तन", "गंभीर", "है"]
    assert tokenize("Global-warming, REAL!") == ["global", "warming", "real"]
    # script ID
    assert script_id("जलवायु परिवर्तन गंभीर है") == "hi"
    assert script_id("জলবায়ু পরিবর্তন গুরুতর") == "bn"
    assert script_id("Climate change bahut serious hai") == "latin"
    assert script_id("Climate change is real भाई, अब तो बारिश का भी कोई pattern नहीं रहा।") == "mixed"
    assert script_id("🙏🌍 12:45") == "other"
    print("preprocess.py self-check passed")
