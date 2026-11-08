import sys
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

MODEL_DIR, DATA_CSV, OUT_CSV = sys.argv[1], sys.argv[2], sys.argv[3]
T = 30  # number of MC Dropout passes

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

with open(f"{MODEL_DIR}/label_map.json") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}
label2id = {v: k for k, v in id2label.items()}

df = pd.read_csv(DATA_CSV)
texts = df["text"].tolist()
y = df["label"].map(label2id).values


def enable_dropout(m):
    m.eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.train()


@torch.no_grad()
def predict_probs(mc, bs=32):
    if mc:
        enable_dropout(model)
    else:
        model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=128,
                  padding=True, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.concatenate(out)


def ece(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(1)


p_det = predict_probs(mc=False)
p_mc = np.mean([predict_probs(mc=True) for _ in range(T)], axis=0)

for name, p in [("Deterministic", p_det), ("MC Dropout", p_mc)]:
    pred = p.argmax(1)
    print(f"{name}: acc={accuracy_score(y, pred):.4f} "
          f"f1={f1_score(y, pred, average='macro'):.4f} "
          f"ECE={ece(p, y):.4f} mean_conf={p.max(1).mean():.4f}")

out = pd.DataFrame({
    "text": texts,
    "true": df["label"],
    "pred_det": [id2label[i] for i in p_det.argmax(1)],
    "conf_det": p_det.max(1),
    "pred_mc": [id2label[i] for i in p_mc.argmax(1)],
    "conf_mc": p_mc.max(1),
    "entropy_mc": entropy(p_mc),
})
out.to_csv(OUT_CSV, index=False)

wrong = out[out["true"] != out["pred_mc"]]
print("\nMisclassified (MC):")
print(wrong[["true", "pred_mc", "conf_mc", "entropy_mc"]])
print("\n10 least confident (MC):")
print(out.nsmallest(10, "conf_mc")[["true", "pred_mc", "conf_mc", "entropy_mc"]])