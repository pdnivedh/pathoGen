import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42

# Load and clean
df = pd.read_csv("data/Symptom2Disease.csv")
df = df.drop(columns=["Unnamed: 0"])
df["label"] = df["label"].str.strip().str.lower()
df["text"] = df["text"].str.strip()

print("Duplicate texts:", df["text"].duplicated().sum())
df = df.drop_duplicates(subset="text").reset_index(drop=True)

# Stratified 70/15/15 split
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
)

# Save splits so every later model uses the same ones
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
print("Split sizes:", len(train_df), len(val_df), len(test_df))

# TF-IDF + logistic regression baseline
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
    LogisticRegression(max_iter=1000),
)
model.fit(train_df["text"], train_df["label"])

pred = model.predict(val_df["text"])
print("Val accuracy:", accuracy_score(val_df["label"], pred))
print("Val macro-F1:", f1_score(val_df["label"], pred, average="macro"))
print(classification_report(val_df["label"], pred))


from sklearn.metrics import confusion_matrix
import numpy as np

labels = sorted(df["label"].unique())
cm = confusion_matrix(val_df["label"], pred, labels=labels)
for i, true_label in enumerate(labels):
    for j, pred_label in enumerate(labels):
        if i != j and cm[i, j] > 0:
            print(f"{true_label} -> {pred_label}: {cm[i, j]}")

wrong = val_df[(val_df["label"] == "diabetes") & (pred != val_df["label"])]
for text, p in zip(wrong["text"], pred[wrong.index.map(val_df.index.get_loc)]):
    print(f"Predicted: {p}\n{text}\n")