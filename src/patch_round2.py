"""Round-2 patch for the pathoGen repo. Run from the repo root (the folder that
contains src/):

    python patch_round2.py

It is safe to run more than once. It fixes src/safety.py (drug classes, see-a-doctor
pattern, the PERSONA pattern that wrongly matched words like "drink" and "dry",
extra drug names, and "Dr." sentence splitting) and replaces the generate()
prompt in src/pipeline.py. Then it checks the syntax and runs four tests.
"""
import re
import sys
import ast
import pathlib


def patch(path, fn):
    p = pathlib.Path(path)
    if not p.exists():
        print("MISSING FILE:", path, "(run this from the repo root)")
        sys.exit(1)
    s = p.read_text(encoding="utf-8")
    s2 = fn(s)
    if s2 == s:
        print("NO CHANGE:", path, "(already patched, or pattern not found)")
    else:
        p.write_text(s2, encoding="utf-8")
        print("patched  :", path)


NEW_DRUG_CLASS = r'''DRUG_CLASS = re.compile(
    r"\b(corticosteroid|steroid|calcineurin inhibitor|anti-inflammatory|"
    r"analgesic|antihistamine|antibiotic|antifungal|antiviral|antimalarial|"
    r"antidiarrheal|painkiller|nsaid|antacid|h2 blocker|proton pump inhibitor|"
    r"ppi|decongestant|antipyretic|laxative|cough suppressant|expectorant|"
    r"retinoid|supplement)s?\b", re.I)'''

NEW_SEE_DOCTOR = r'''SEE_DOCTOR = re.compile(
    r"(see|consult|contact|visit|seek).{0,40}(doctor|physician|healthcare|"
    r"medical|dermatologist|specialist|neurologist|gastroenterologist)|"
    r"medical attention", re.I)'''

NEW_PERSONA = r'''PERSONA = re.compile(
    r"(i have read your query|i (would )?(suggest|recommend|advise)|"
    r"in my opinion|i think|\bdr\.?\s+[a-z]+)", re.I)'''

DR_LINE = '    text = re.sub(r"\\b(Dr|Mr|Mrs|Ms)\\.", r"\\1", text)\n'

NEW_GENERATE = r'''@torch.no_grad()
def generate(disease, symptoms, hits):
    notes = "\n".join(f"- {r}" for _, _, r in hits)
    messages = [
        {"role": "system", "content":
            "You are a careful medical information assistant, not a doctor. "
            "Use only the doctor notes provided, which were written for other "
            "patients. Never name medicines, doses, drug classes or people. "
            "Never assume the patient is already taking any treatment. "
            "Reply in at most 3 plain sentences with no lists and no markdown, "
            "and say when to see a doctor."},
        {"role": "user", "content":
            f"Doctor notes from similar cases:\n{notes}\n\n"
            f"Patient symptoms: {symptoms}\n"
            f"Working condition: {disease}. Do not suggest a different diagnosis.\n\n"
            "Give general self-care advice."},
    ]
    text = ch_tok.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True)
    enc = ch_tok(text, return_tensors="pt").to(device)
    out = ch.generate(**enc, max_new_tokens=130, do_sample=False,
                      repetition_penalty=1.1)
    return ch_tok.decode(out[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True)'''


def fix_safety(s):
    s = re.sub(r"DRUG_CLASS = re\.compile\(.*?re\.I\)", lambda m: NEW_DRUG_CLASS,
               s, count=1, flags=re.S)
    s = re.sub(r"SEE_DOCTOR = re\.compile\(.*?re\.I\)", lambda m: NEW_SEE_DOCTOR,
               s, count=1, flags=re.S)
    s = re.sub(r"PERSONA = re\.compile\(.*?re\.I\)", lambda m: NEW_PERSONA,
               s, count=1, flags=re.S)
    if '"ketoconazole"' not in s:
        s = s.replace('"ranitidine"]',
                      '"ranitidine",\n              "ketoconazole", "clotrimazole", "dettol"]', 1)
    if "(Dr|Mr|Mrs|Ms)" not in s:
        s = s.replace("def check(text, disease):\n",
                      "def check(text, disease):\n" + DR_LINE, 1)
    return s


def fix_pipeline(s):
    return re.sub(r"@torch\.no_grad\(\)\s*\ndef generate\(.*?(?=\ndef tidy)",
                  lambda m: NEW_GENERATE + "\n\n", s, count=1, flags=re.S)


patch("src/safety.py", fix_safety)
patch("src/pipeline.py", fix_pipeline)

for f in ("src/safety.py", "src/pipeline.py"):
    ast.parse(pathlib.Path(f).read_text(encoding="utf-8"))
print("syntax OK")

sys.path.insert(0, "src")
import safety  # noqa: E402

tests = [
    ("Drink plenty of fluids and keep the skin dry.", "fungal infection", False),
    ("Continue using antihistamines as prescribed by Dr. Patel.", "fungal infection", True),
    ("Over-the-counter antimalarial medications may help.", "malaria", True),
    ("Rest, drink fluids and see a dermatologist if it worsens.", "psoriasis", False),
]
bad = 0
for text, disease, should_flag in tests:
    got = bool(safety.check_sentence(text, disease))
    ok = got == should_flag
    bad += (not ok)
    print("OK  " if ok else "FAIL", text)
print("ALL TESTS PASSED" if bad == 0 else f"{bad} TEST(S) FAILED - paste this output to Claude")
