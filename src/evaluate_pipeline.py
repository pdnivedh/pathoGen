"""Evaluate a PathoGen pipeline output CSV.

Usage (from the pathoGen repo folder, venv active):
    python src/evaluate_pipeline.py <pipeline_csv> [--no-embed]

Reports: classifier + review-flag stats, a keyword-based retrieval-precision
PROXY (not human relevance judgements), embedding similarity between the final
text and its retrieved evidence (with a shuffled-evidence control), and safety
filter statistics. Writes <pipeline_csv>.scored.csv and <pipeline_csv>.metrics.json
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from safety import check  # noqa: E402

# disease -> (strict names, extra lenient keywords). Lowercase substring match.
ALIASES = {
    "acne": (["acne"], ["pimple", "comedone"]),
    "allergy": (["allerg"], ["rhinitis", "hives", "urticaria", "antihistamin"]),
    "arthritis": (["arthritis"], ["rheumat", "joint", "gout"]),
    "bronchial asthma": (["asthma"], ["wheez", "inhaler", "bronchodilator"]),
    "cervical spondylosis": (["spondyl"], ["cervical", "neck", "disc"]),
    "chicken pox": (["chicken pox", "chickenpox", "varicella"], ["pox"]),
    "common cold": (["common cold"], ["cold", "flu", "viral", "sore throat", "runny nose"]),
    "dengue": (["dengue"], ["platelet"]),
    "diabetes": (["diabet"], ["sugar", "glucose", "insulin"]),
    "dimorphic hemorrhoids": (["hemorrhoid", "haemorrhoid", "piles"], ["fissure", "stool"]),
    "drug reaction": (["drug reaction", "drug allergy"], ["medication", "side effect", "reaction"]),
    "fungal infection": (["fung", "candid", "ringworm", "tinea"], ["antifungal"]),
    "gastroesophageal reflux disease": (["gerd", "reflux"], ["acidity", "heartburn", "esophag", "acid"]),
    "hypertension": (["hypertens"], ["blood pressure"]),
    "impetigo": (["impetigo"], ["crust", "sores", "bacterial skin"]),
    "jaundice": (["jaundice"], ["bilirubin", "liver", "hepatitis"]),
    "malaria": (["malaria"], ["plasmodium", "mosquito"]),
    "migraine": (["migraine"], ["headache"]),
    "peptic ulcer disease": (["ulcer"], ["pylori", "gastritis", "stomach"]),
    "pneumonia": (["pneumonia"], ["chest infection", "lung"]),
    "psoriasis": (["psoria"], ["scaly", "scaling"]),
    "typhoid": (["typhoid"], ["widal", "enteric"]),
    "urinary tract infection": (["urinary", "uti"], ["urine", "bladder"]),
    "varicose veins": (["varicose"], ["vein", "compression"]),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / max(n, 1):.1%} (95% CI {lo:.0%}-{hi:.0%})"


def snippet_matches(snippet, disease):
    strict, extra = ALIASES.get(disease, ([disease], []))
    s = snippet.lower()
    return (any(a in s for a in strict), any(a in s for a in strict + extra))


def retrieval_proxy(df):
    strict_hits, lenient_hits, totals = 0, 0, 0
    hit_strict, hit_lenient = 0, 0
    for ev, pred in zip(df["evidence"], df["pred"]):
        snippets = [x for x in ev.split(" || ") if x.strip()]
        if not snippets:
            continue
        flags = [snippet_matches(x, pred) for x in snippets]
        strict_hits += sum(f[0] for f in flags)
        lenient_hits += sum(f[1] for f in flags)
        totals += len(flags)
        hit_strict += int(any(f[0] for f in flags))
        hit_lenient += int(any(f[1] for f in flags))
    return strict_hits, lenient_hits, totals, hit_strict, hit_lenient


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    use_embed = "--no-embed" not in sys.argv

    df = pd.read_csv(path)
    for col in ["evidence", "raw_text", "safe_text", "flag_codes"]:
        df[col] = df[col].fillna("")
    n_all = len(df)
    out = {"file": os.path.basename(path), "cases": n_all}

    # 1. Classifier and review flag
    correct = df["true"] == df["pred"]
    flagged = df["flagged"].astype(bool)
    errors = ~correct
    out["accuracy"] = float(correct.mean())
    out["flagged_for_review"] = int(flagged.sum())
    out["errors_total"] = int(errors.sum())
    out["errors_caught_by_flag"] = int((errors & flagged).sum())
    out["accuracy_on_unflagged"] = float(correct[~flagged].mean()) if (~flagged).any() else None
    print(f"\nCases: {n_all}")
    print(f"Classifier accuracy: {fmt(int(correct.sum()), n_all)}")
    print(f"Flagged for review (conf < 0.80): {int(flagged.sum())} ({flagged.mean():.1%})")
    print(f"Errors: {int(errors.sum())}, caught by the review flag: {int((errors & flagged).sum())}")
    if (~flagged).any():
        print(f"Accuracy on unflagged cases: {fmt(int(correct[~flagged].sum()), int((~flagged).sum()))}")

    # 2. Generated cases only
    g = df[(~flagged) & (df["evidence"] != "")].copy().reset_index(drop=True)
    ng = len(g)
    out["generated"] = ng
    print(f"\nGenerated suggestions: {ng}")
    if ng == 0:
        return

    # 3. Retrieval precision proxy
    sh, lh, tot, hs, hl = retrieval_proxy(g)
    out["retrieval_precision_strict"] = sh / tot
    out["retrieval_precision_lenient"] = lh / tot
    out["retrieval_hit_at3_strict"] = hs / ng
    out["retrieval_hit_at3_lenient"] = hl / ng
    print("\nRetrieval PROXY (keyword match of the predicted disease in the retrieved doctor notes;")
    print("not human relevance judgements)")
    print(f"  precision@3, strict names only : {fmt(sh, tot)}")
    print(f"  precision@3, lenient keywords  : {fmt(lh, tot)}")
    print(f"  at least 1 of 3 relevant, strict : {fmt(hs, ng)}")
    print(f"  at least 1 of 3 relevant, lenient: {fmt(hl, ng)}")

    per = {}
    for d in sorted(g["pred"].unique()):
        sub = g[g["pred"] == d]
        a, b, t, _, _ = retrieval_proxy(sub)
        per[d] = (a, t)
    weak = sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])[:6]
    print("  weakest diseases (strict precision):",
          ", ".join(f"{d} {a}/{t}" for d, (a, t) in weak))

    # 4. Safety filter statistics
    codes = {}
    for s in g["flag_codes"]:
        for c in [x for x in s.split(",") if x]:
            codes[c] = codes.get(c, 0) + 1
    any_flag = int((g["flag_codes"] != "").sum())
    high = int(g["unsafe_raw"].astype(bool).sum())
    fb = int(g["fallback"].astype(bool).sum())
    residual = 0
    for text, pred in zip(g["safe_text"], g["pred"]):
        if check(text, pred)["flags"]:
            residual += 1
    out.update({"raw_any_flag": any_flag, "raw_high_severity": high,
                "fallback": fb, "residual_flags_after_filter": residual,
                "flag_code_counts": codes})
    print("\nSafety filter (automatic rules; a floor, not the true unsafe rate)")
    print(f"  raw outputs with any flag      : {fmt(any_flag, ng)}")
    print(f"  raw outputs with high severity : {fmt(high, ng)}")
    print(f"  replaced by the fallback text  : {fmt(fb, ng)}")
    print(f"  final text still triggering a rule: {fmt(residual, ng)}")
    print(f"  flag code counts: {codes}")

    # 5. Embedding similarity with a shuffled-evidence control
    if use_embed:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        model.max_seq_length = 256

        def enc(texts):
            return model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                                show_progress_bar=False)

        ev = g["evidence"].str.replace(" || ", " ", regex=False)
        e_safe, e_raw = enc(g["safe_text"]), enc(g["raw_text"])
        e_ev, e_sym = enc(ev), enc(g["text"])
        rng = np.random.default_rng(42)
        perm = rng.permutation(ng)
        for i in range(ng):          # make sure no case is paired with itself
            if perm[i] == i:
                perm[i] = (i + 1) % ng
        g["sim_safe_evidence"] = (e_safe * e_ev).sum(1)
        g["sim_raw_evidence"] = (e_raw * e_ev).sum(1)
        g["sim_safe_shuffled"] = (e_safe * e_ev[perm]).sum(1)
        g["sim_safe_symptoms"] = (e_safe * e_sym).sum(1)
        wins = int((g["sim_safe_evidence"] > g["sim_safe_shuffled"]).sum())
        out.update({
            "sim_safe_evidence_mean": float(g["sim_safe_evidence"].mean()),
            "sim_raw_evidence_mean": float(g["sim_raw_evidence"].mean()),
            "sim_safe_shuffled_mean": float(g["sim_safe_shuffled"].mean()),
            "sim_safe_symptoms_mean": float(g["sim_safe_symptoms"].mean()),
            "own_evidence_beats_shuffled": wins,
        })
        print("\nEmbedding similarity (all-MiniLM-L6-v2 cosine; first 256 tokens of each text)")
        print(f"  final text vs its own evidence   : {g['sim_safe_evidence'].mean():.3f}")
        print(f"  raw text vs its own evidence     : {g['sim_raw_evidence'].mean():.3f}")
        print(f"  final text vs shuffled evidence  : {g['sim_safe_shuffled'].mean():.3f}  (control)")
        print(f"  final text vs patient symptoms   : {g['sim_safe_symptoms'].mean():.3f}")
        print(f"  own evidence closer than shuffled: {fmt(wins, ng)}")

    g.to_csv(path + ".scored.csv", index=False)
    with open(path + ".metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {path}.scored.csv and {path}.metrics.json")


if __name__ == "__main__":
    main()