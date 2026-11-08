import sys, os, re
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

CSV, OUT_DIR = sys.argv[1], sys.argv[2]
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV).dropna().reset_index(drop=True)

def fix_spacing(t):
    # "sperms.Consult" -> "sperms. Consult" (only lowercase + punct + uppercase)
    return re.sub(r"([a-z][.,!?])([A-Z])", r"\1 \2", t)

df["doctor_response"] = df["doctor_response"].map(fix_spacing)
df.to_csv(f"{OUT_DIR}/corpus.csv", index=False)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.max_seq_length = 256

emb = model.encode(
    df["patient_message"].tolist(),
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True,
).astype("float32")

index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine on normalized vectors
index.add(emb)
faiss.write_index(index, f"{OUT_DIR}/index.faiss")
print("Indexed:", index.ntotal, "vectors, dim", emb.shape[1])

# Quick sanity test
queries = [
    "I have been experiencing a skin rash on my arms, legs, and torso. It is red, itchy, and covered in dry, scaly patches.",
    "I have a high fever, severe headache, and pain behind my eyes with joint pain.",
]
for q in queries:
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, 3)
    print("\nQUERY:", q)
    for s, i in zip(scores[0], ids[0]):
        print(f"  [{s:.3f}] {df['patient_message'][i][:150]}")