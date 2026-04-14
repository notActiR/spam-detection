"""
Preprocess: download Enron-Spam dataset and clean emails.
"""
import os, re, urllib.request, tarfile, pickle, ssl
import nltk

ssl._create_default_https_context = ssl._create_unverified_context
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

URLS = [
    "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz",
    "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron2.tar.gz",
    "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron3.tar.gz",
]
DATA_DIR = "data"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for url in URLS:
        fname = os.path.join(DATA_DIR, url.split("/")[-1])
        if not os.path.exists(fname):
            print(f"Downloading {url.split('/')[-1]}...")
            urllib.request.urlretrieve(url, fname)
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(DATA_DIR)


def clean(text):
    text = re.sub(r'<[^>]+>', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(t for t in text.split() if t not in STOP_WORDS and len(t) > 2)


def load():
    texts, labels = [], []
    for folder in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(fp):
            continue
        for cat, label in [("ham", 0), ("spam", 1)]:
            cp = os.path.join(fp, cat)
            if not os.path.isdir(cp):
                continue
            for fname in os.listdir(cp):
                try:
                    with open(os.path.join(cp, fname), 'r', encoding='utf-8', errors='ignore') as f:
                        texts.append(clean(f.read()))
                        labels.append(label)
                except Exception:
                    pass
    return texts, labels


if __name__ == "__main__":
    download()
    texts, labels = load()
    print(f"Total: {len(texts)} | Spam: {sum(labels)} | Ham: {len(labels)-sum(labels)}")
    with open(os.path.join(DATA_DIR, "processed.pkl"), "wb") as f:
        pickle.dump((texts, labels), f)
    print("Saved to data/processed.pkl")
