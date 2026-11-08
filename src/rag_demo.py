import sys, re
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

RET_DIR = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"

corpus = pd.read_csv(f"{RET_DIR}/corpus.csv")
index = faiss.read_index(f"{RET_DIR}/index.faiss")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

GEN = "google/flan-t5-large"
tok = AutoTokenizer.from_pretrained(GEN)
gen = AutoModelForSeq2SeqLM.from_pretrained(GEN).to(device)


def retrieve(query, k=3):
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, k)
    return [(float(s), int(i)) for s, i in zip(scores[0], ids[0])]


def clean(t, n=400):
    return re.sub(r"\s+", " ", t).strip()[:n]


def build_prompt(disease, symptoms, hits):
    evidence = "\n".join(f"- {clean(corpus['doctor_response'][i])}" for _, i in hits)
    return (
        f"A patient reports: {symptoms}\n"
        f"Likely condition: {disease}.\n"
        f"Doctor responses to similar cases:\n{evidence}\n\n"
        "Using only the doctor responses above, write a short, general treatment "
        "suggestion for this patient and say when to see a doctor."
    )


@torch.no_grad()
def generate(prompt):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = gen.generate(**enc, max_new_tokens=150, num_beams=4, no_repeat_ngram_size=3)
    return tok.decode(out[0], skip_special_tokens=True)


cases = [
    ("psoriasis",
     "I have been experiencing a skin rash on my arms, legs, and torso. "
     "It is red, itchy, and covered in dry, scaly patches."),
    ("dengue",
     "I have a high fever, severe headache, and pain behind my eyes with joint pain."),
]

for disease, symptoms in cases:
    print("=" * 80)
    print("SYMPTOMS:", symptoms)
    print("DISEASE :", disease)

    plain = retrieve(symptoms)
    aug = retrieve(f"{disease}. {symptoms}")
    print("\nTop retrieved, symptoms only:")
    for s, i in plain:
        print(f"  [{s:.3f}] {clean(corpus['patient_message'][i], 110)}")
    print("Top retrieved, disease + symptoms:")
    for s, i in aug:
        print(f"  [{s:.3f}] {clean(corpus['patient_message'][i], 110)}")

    prompt = build_prompt(disease, symptoms, aug)
    print("\nGENERATED:\n", generate(prompt))