import sys
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score

MODEL = "dmis-lab/biobert-base-cased-v1.2"
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "results/biobert"
SEED = 42
set_seed(SEED)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
val = pd.read_csv(f"{DATA_DIR}/val.csv")

labels = sorted(train["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
for df in (train, val):
    df["labels"] = df["label"].map(label2id)

tok = AutoTokenizer.from_pretrained(MODEL)

def prep(df):
    ds = Dataset.from_pandas(df[["text", "labels"]])
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

train_ds, val_ds = prep(train), prep(val)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(labels)
)

print("Classifier outputs:", model.classifier.out_features)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="macro"),
    }

args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=6,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    warmup_steps=30,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)

trainer.train()
print("Final val metrics:", trainer.evaluate())

trainer.save_model(f"{OUT_DIR}/best")
tok.save_pretrained(f"{OUT_DIR}/best")
with open(f"{OUT_DIR}/best/label_map.json", "w") as f:
    json.dump(id2label, f)