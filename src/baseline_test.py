"""TF-IDF + logistic regression baseline, evaluated on the saved val AND test splits.

Run from the repo root (the folder that contains data/ and src/):
    python src/baseline_test.py

It reads data/train.csv, data/val.csv and data/test.csv (it does not re-split).
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
    LogisticRegression(max_iter=1000),
)
model.fit(train["text"], train["label"])

for name, df in [("val", val), ("test", test)]:
    pred = model.predict(df["text"])
    acc = accuracy_score(df["label"], pred)
    f1 = f1_score(df["label"], pred, average="macro")
    errors = int((pred != df["label"]).sum())
    print(f"{name}: n={len(df)}  accuracy={acc:.4f}  macro-F1={f1:.4f}  errors={errors}")
