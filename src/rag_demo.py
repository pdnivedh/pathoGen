import sys, re
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

RET_DIR = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"

corpus = pd.read_csv(f"{RET_DIR}/corpus.csv")
index = faiss.read_index(f"{RET_DIR}/index.faiss")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Generator A: FLAN-T5 (proposal baseline)
t5_tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)

# Generator B: instruction-tuned chat model (comparison)
CHAT = "Qwen/Qwen2.5-1.5B-Instruct"
ch_tok = AutoTokenizer.from_pretrained(CHAT)
ch = AutoModelForCausalLM.from_pretrained(CHAT, dtype=torch.float16).to(device)

BOILER = re.compile(
    r"(hope (i|this|that)|regards|let me know|thank|dr\.|best wishes|take care|"
    r"feel free|revert|further (query|question)|welcome to|i have gone through|"
    r"online|chat|upload|picture|photo|\bsite\b|report section|consult)", re.I)
DOSE = re.compile(
    r"(\d+\s?(mg|ml|mcg|gm|g)\b|tablet|\btab\b|capsule|syrup|injection|ointment|"
    r"twice daily|once daily|thrice|times a day|times daily|\bbd\b|\bod\b|\btds\b)", re.I)


def clean_response(t):
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(hi|hello|dear)[^.,!]*[,.!]\s*", "", t, flags=re.I)
    sents = re.split(r"(?<=[.!?])\s+", t)
    keep = [s for s in sents
            if not BOILER.search(s) and not DOSE.search(s) and not s.endswith("?")]
    return " ".join(keep).strip()


def short(t, n=130):
    return re.sub(r"\s+", " ", t).strip()[:n]


def retrieve(query, k=3, pool=15, min_chars=120):
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


def notes_block(hits):
    return "\n".join(f"- {r}" for _, _, r in hits)


@torch.no_grad()
def gen_t5(disease, symptoms, hits):
    prompt = (
        f"Doctor notes from similar cases:\n{notes_block(hits)}\n\n"
        f"Patient symptoms: {symptoms}\nLikely condition: {disease}\n\n"
        "Write a general treatment suggestion in 2 to 3 sentences based on the "
        "doctor notes, and say when the patient should see a doctor:"
    )
    enc = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = t5.generate(**enc, max_new_tokens=120, min_new_tokens=30,
                      num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.3)
    return t5_tok.decode(out[0], skip_special_tokens=True)


@torch.no_grad()
def gen_chat(disease, symptoms, hits):
    messages = [
        {"role": "system", "content":
            "You are a careful medical information assistant, not a doctor. "
            "Use only the doctor notes provided. Do not name medicines or give doses. "
            "Write 2 to 3 plain sentences."},
        {"role": "user", "content":
            f"Doctor notes from similar cases:\n{notes_block(hits)}\n\n"
            f"Patient symptoms: {symptoms}\nLikely condition (from a classifier, may be wrong): {disease}\n\n"
            "Give general self-care advice and say when the patient should see a doctor."},
    ]
    text = ch_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = ch_tok(text, return_tensors="pt").to(device)
    out = ch.generate(**enc, max_new_tokens=150, do_sample=False, repetition_penalty=1.1)
    return ch_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


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
    print("\nCleaned evidence:")
    for s, i, r in hits:
        print(f"  [{s:.3f}] {short(r)}")
    print("\nFLAN-T5-large:\n", gen_t5(disease, symptoms, hits))
    print("\nQwen2.5-1.5B-Instruct:\n", gen_chat(disease, symptoms, hits))