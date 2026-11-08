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

BOILER = re.compile(
    r"(hope (i|this|that)|regards|let me know|thank|dr\.|best wishes|take care|"
    r"feel free|revert|further (query|question)|welcome to|i have gone through|"
    r"online|chat)", re.I)


def clean_response(t):
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(hi|hello|dear)[^.,!]*[,.!]\s*", "", t, flags=re.I)
    sents = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(s for s in sents if not BOILER.search(s)).strip()


def short(t, n=110):
    return re.sub(r"\s+", " ", t).strip()[:n]


def retrieve(query, k=3, pool=10, min_chars=120):
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, pool)
    hits = []
    for s, i in zip(scores[0], ids[0]):
        resp = clean_response(corpus["doctor_response"][int(i)])
        if len(resp) >= min_chars:
            hits.append((float(s), int(i), resp[:450]))
        if len(hits) == k:
            break
    return hits


def build_prompt(disease, symptoms, hits):
    notes = "\n".join(f"- {r}" for _, _, r in hits)
    return (
        "Doctor notes from similar cases:\n"
        f"{notes}\n\n"
        f"Patient symptoms: {symptoms}\n"
        f"Likely condition: {disease}\n\n"
        "Write a general treatment suggestion in 2 to 3 sentences based on the "
        "doctor notes, and say when the patient should see a doctor:"
    )


@torch.no_grad()
def generate(prompt):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = gen.generate(
        **enc, max_new_tokens=120, min_new_tokens=30,
        num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.3,
    )
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
    hits = retrieve(f"{disease}. {symptoms}")
    print("\nCleaned evidence used:")
    for s, i, r in hits:
        print(f"  [{s:.3f}] {short(r, 160)}")
    print("\nGENERATED:\n", generate(build_prompt(disease, symptoms, hits)))