import sys
import re
import json
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM)
from safety import check, FALLBACK

CLS_DIR, RET_DIR, CASES_CSV, OUT_CSV = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 40
THRESHOLD = 0.80
REVIEW_MSG = ("Low confidence in the automatic assessment. This case has been "
              "flagged for expert review.")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Stage 1: classifier
cls_tok = AutoTokenizer.from_pretrained(CLS_DIR)
cls = AutoModelForSequenceClassification.from_pretrained(CLS_DIR).to(device).eval()
with open(f"{CLS_DIR}/label_map.json") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

# Stage 2: retrieval
corpus = pd.read_csv(f"{RET_DIR}/corpus.csv")
index = faiss.read_index(f"{RET_DIR}/index.faiss")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Stage 3: generator
CHAT = "Qwen/Qwen2.5-1.5B-Instruct"
ch_tok = AutoTokenizer.from_pretrained(CHAT)
ch = AutoModelForCausalLM.from_pretrained(CHAT, dtype=torch.float16).to(device)

BOILER = re.compile(
    r"(hope (i|this|that)|regards|let me know|thank|dr\.|best wishes|take care|"
    r"feel free|revert|further (query|question)|welcome to|i have gone through|"
    r"online|chat|upload|picture|photo|attachment|\bsite\b|report section|"
    r"consult)", re.I)
DOSE_EV = re.compile(
    r"(\d+\s?(mg|ml|mcg|gm|g)\b|tablet|\btab\b|capsule|syrup|injection|"
    r"ointment|twice daily|once daily|thrice|times a day|times daily|"
    r"\ba day\b|\bbd\b|\bod\b|\btds\b)", re.I)


@torch.no_grad()
def classify(text):
    enc = cls_tok(text, truncation=True, max_length=128,
                  return_tensors="pt").to(device)
    probs = torch.softmax(cls(**enc).logits, dim=-1)[0]
    conf, idx = probs.max(dim=0)
    return id2label[int(idx)], float(conf)


def clean_response(t):
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(hi|hello|dear)[^.,!]*[,.!]\s*", "", t, flags=re.I)
    sents = re.split(r"(?<=[.!?])\s+", t)
    keep = [s for s in sents
            if not BOILER.search(s) and not DOSE_EV.search(s)
            and not s.endswith("?")]
    return " ".join(keep).strip()


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


@torch.no_grad()
def generate(disease, symptoms, hits):
    notes = "\n".join(f"- {r}" for _, _, r in hits)
    messages = [
        {"role": "system", "content":
            "You are a careful medical information assistant, not a doctor. "
            "Use only the doctor notes provided. Never name medicines, doses "
            "or drug classes. Reply in at most 3 plain sentences with no lists "
            "and no markdown, and say when to see a doctor."},
        {"role": "user", "content":
            f"Doctor notes from similar cases:\n{notes}\n\n"
            f"Patient symptoms: {symptoms}\n"
            f"Likely condition (from a classifier, may be wrong): {disease}\n\n"
            "Give general self-care advice."},
    ]
    text = ch_tok.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True)
    enc = ch_tok(text, return_tensors="pt").to(device)
    out = ch.generate(**enc, max_new_tokens=110, do_sample=False,
                      repetition_penalty=1.1)
    return ch_tok.decode(out[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True)


def tidy(t, max_sents=4):
    t = re.sub(r"[*#]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    sents = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(sents[:max_sents])


cases = pd.read_csv(CASES_CSV).sample(n=N, random_state=42).reset_index(drop=True)
rows = []

for k, r in cases.iterrows():
    symptoms = r["text"]
    pred, conf = classify(symptoms)
    row = {"text": symptoms, "true": r["label"], "pred": pred,
           "conf": round(conf, 4), "flagged": conf < THRESHOLD,
           "evidence": "", "raw_text": "", "safe_text": "",
           "unsafe_raw": False, "fallback": False, "flag_codes": ""}

    if row["flagged"]:
        row["safe_text"] = REVIEW_MSG
    else:
        hits = retrieve(f"{pred}. {symptoms}")
        if not hits:
            row["safe_text"] = FALLBACK
            row["fallback"] = True
        else:
            raw = tidy(generate(pred, symptoms, hits))
            res = check(raw, pred)
            row["evidence"] = " || ".join(h[2] for h in hits)
            row["raw_text"] = raw
            row["safe_text"] = res["safe_text"]
            row["unsafe_raw"] = res["unsafe"]
            row["fallback"] = res["fallback"]
            row["flag_codes"] = ",".join(sorted({c for c, _, _ in res["flags"]}))
    rows.append(row)
    print(f"[{k + 1}/{len(cases)}] true={row['true']} pred={pred} conf={conf:.2f}"
          f"{' FLAGGED' if row['flagged'] else ''}")

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)

gen = out[~out["flagged"]]
print("\n" + "=" * 60)
print(f"Cases: {len(out)} | flagged for review: {int(out['flagged'].sum())}")
print(f"Classifier accuracy (all cases): {(out['true'] == out['pred']).mean():.3f}")
if len(gen):
    print(f"Generated: {len(gen)}")
    print(f"Raw outputs with a high-severity violation: {gen['unsafe_raw'].mean():.1%}")
    print(f"Outputs replaced by fallback message: {gen['fallback'].mean():.1%}")
    print("\nExamples:")
    for _, e in gen.head(3).iterrows():
        print(f"\nSYMPTOMS: {e['text'][:140]}")
        print(f"PRED: {e['pred']} (conf {e['conf']})  FLAGS: {e['flag_codes'] or 'none'}")
        print(f"RAW : {e['raw_text']}")
        print(f"SAFE: {e['safe_text']}")