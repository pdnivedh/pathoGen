import pandas as pd

df = pd.read_csv("data/Symptom2Disease.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print()
print(df.head())
print()
print("Missing values:")
print(df.isna().sum())
print()
print("Number of diseases:", df["label"].nunique())
print(df["label"].value_counts())
print()
print("Example text:")
print(df["text"].iloc[0])