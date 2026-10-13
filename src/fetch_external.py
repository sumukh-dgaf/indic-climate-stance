"""Download the public external sources and normalise them into data/external/*.csv.

Schema of every normalised file: id,comment,stance,language,target,domain,source,label_original
Raw downloads go to data/raw/ (git-ignored: third-party licences, re-fetchable).
Run:  python src/fetch_external.py        (needs: pip install datasets huggingface_hub pandas)
"""
import json, os, re, urllib.parse, urllib.request, zipfile
import pandas as pd

RAW, OUT = "data/raw/", "data/external/"
COLS = ["id", "comment", "stance", "language", "target", "domain", "source", "label_original"]
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)


def http(url):
    return urllib.request.urlopen(url, timeout=90).read()


def save(df, name):
    df = df[COLS]
    assert df.stance.isin(["Favour", "Against", "None"]).all(), "unmapped label"
    df.to_csv(OUT + name, index=False)
    print(f"{name:42s} rows={len(df):5d}  {df.stance.value_counts().to_dict()}")


def tweet_eval():  # SemEval-2016 Task 6, climate target, via TweetEval (label ids: 0 none, 1 against, 2 favor)
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "stance_climate")
    names = ds["train"].features["label"].names
    m = {"none": "None", "against": "Against", "favor": "Favour"}
    rows = []
    for split in ds:
        for i, r in enumerate(ds[split]):
            lab = names[r["label"]]
            rows.append([f"tweeteval_{split}_{i}", r["text"], m[lab], "en", "Climate Change is a Real Concern",
                         "tweets", "SemEval-2016 Task 6 via TweetEval", lab])
    save(pd.DataFrame(rows, columns=COLS), "semeval2016_climate_en.csv")


def dhoroni():  # Bengali NEWS ARTICLES (not comments). Problematic (107) has no mapping -> dropped, and counted.
    from huggingface_hub import hf_hub_download
    d = RAW + "dhoroni/"
    csv = hf_hub_download("ciol-research/Dhoroni", "dataset.csv", repo_type="dataset", local_dir=d)
    z = hf_hub_download("ciol-research/Dhoroni", "dataset_files.zip", repo_type="dataset", local_dir=d)
    if not os.path.isdir(d + "articles"):
        zipfile.ZipFile(z).extractall(d + "articles")
    df = pd.read_csv(csv, keep_default_na=False)
    m = {"Positive": "Favour", "Negative": "Against", "Neutral": "None"}
    print("Dhoroni dropped (no mapping):", df["Stance Detection"].map(lambda x: x not in m).sum())
    df = df[df["Stance Detection"].isin(m)].copy()
    body = [open(f"{d}articles/dataset_files/{f}", encoding="utf-8", errors="replace").read().strip() for f in df.File]
    df["comment"] = [b if b else t for b, t in zip(body, df.Title)]  # 10 articles have empty body -> headline
    df = df.assign(id="dhoroni_" + df.File.str.replace(".txt", "", regex=False), stance=df["Stance Detection"].map(m),
                   language="bn", target="climate change (news article stance)", domain="news articles (NOT social media)",
                   source="Dhoroni (Wasi et al. 2024)", label_original=df["Stance Detection"])
    save(df, "dhoroni_bn_news.csv")


def multiclimate():  # English video-transcript sentences. 'oppose' = negative sentiment toward CC itself: mapping is LOW-confidence.
    tree = json.loads(http("https://api.github.com/repos/werywjw/MultiClimate/git/trees/HEAD?recursive=1"))["tree"]
    base = [x["path"] for x in tree if re.fullmatch(r"dataset/[^/]+/[^/_]+\.csv", x["path"])]
    os.makedirs(RAW + "multiclimate", exist_ok=True)
    frames = []
    for p in base:
        f = RAW + "multiclimate/" + p.split("/")[1] + ".csv"
        if not os.path.exists(f):
            open(f, "wb").write(http("https://raw.githubusercontent.com/werywjw/MultiClimate/HEAD/" + urllib.parse.quote(p)))
        d = pd.read_csv(f, keep_default_na=False)
        d.columns = [c.strip() for c in d.columns]
        d["comment"] = (d["text.1"] if "text.1" in d.columns else d["text"]).astype(str).str.strip()  # WICC.csv: real text is in 'text.1'
        d["video"] = p.split("/")[1]
        frames.append(d[["video", "label", "comment"]])
    df = pd.concat(frames, ignore_index=True)
    print("MultiClimate rows before dropping empty text:", len(df), "(paper: 4,209)")
    df = df[df.comment != ""].reset_index(drop=True)
    m = {0: ("None", "neutral"), 1: ("Favour", "support"), 2: ("Against", "oppose")}
    df["stance"] = df.label.map(lambda x: m[int(x)][0])
    df["label_original"] = df.label.map(lambda x: m[int(x)][1])
    df = df.assign(id=df.video + "_" + df.index.astype(str), language="en", target="climate change (video stance)",
                   domain="YouTube video transcript sentences (NOT comments)", source="MultiClimate (Wang et al. 2024)")
    save(df, "multiclimate_en_video.csv")


def swami():  # TARGET IS DEMONETISATION, NOT CLIMATE. Same label names, different meaning. Never merge into the climate pool.
    base = "https://raw.githubusercontent.com/sahilswami96/StanceDetection_CodeMixed/HEAD/Dataset/"
    os.makedirs(RAW + "swami", exist_ok=True)
    for f in ["Notebandi_tweets.txt", "Notebandi_tweets_stance.txt"]:
        if not os.path.exists(RAW + "swami/" + f):
            open(RAW + "swami/" + f, "wb").write(http(base + f))
    txt = open(RAW + "swami/Notebandi_tweets.txt", encoding="utf-8").read().replace("\r\n", "\n")
    tw = {}
    for block in re.split(r"\n\s*\n", txt):
        lines = block.strip("\n").split("\n")
        if lines[0].strip():
            tw[lines[0].strip()] = " ".join(x.strip() for x in lines[1:])
    sl = [x.strip() for x in open(RAW + "swami/Notebandi_tweets_stance.txt", encoding="utf-8").read().replace("\r\n", "\n").split("\n") if x.strip()]
    st = dict(zip(sl[0::2], sl[1::2]))  # stance file alternates id / label lines
    m = {"FAVOR": "Favour", "AGAINST": "Against", "NONE": "None"}
    ids = [i for i in tw if i in st]
    df = pd.DataFrame({"id": ["swami_" + i for i in ids], "comment": [tw[i] for i in ids],
                       "stance": [m[st[i]] for i in ids], "label_original": [st[i] for i in ids]})
    df = df.assign(language="hi-Latn (romanised, code-mixed)", target="demonetisation (NOT climate)", domain="tweets",
                   source="Swami et al. 2018 English-Hindi code-mixed")
    assert len(df) == 3545, len(df)  # paper reports 3,545 tweets: 964 favour / 647 against / 1,934 none
    save(df, "swami_hinglish_demonetisation.csv")


if __name__ == "__main__":
    for fn in (tweet_eval, dhoroni, multiclimate, swami):
        fn()
