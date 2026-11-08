from datasets import load_dataset

ds = load_dataset("OpenMed/MedDialog")
print(ds)

train = ds["train"]
print("Columns:", train.column_names)
print("Rows:", len(train))

ex = train[0]
print("\nPatient:", ex["patient_message"])
print("\nDoctor:", ex["doctor_response"])

# Keep a 20k subsample for fast embedding on Colab
df = train.shuffle(seed=42).select(range(20000)).to_pandas()
df = df[["patient_message", "doctor_response"]].reset_index(drop=True)
df.to_csv("data/dialogues_20k.csv", index=False)
print("\nSaved data/dialogues_20k.csv", df.shape)