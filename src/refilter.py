"""Re-apply the safety filter to an existing pipeline CSV.

Generation does not depend on the filter (retrieval -> generation -> filter),
so after fixing a filter rule the saved raw_text can simply be filtered again.

Usage (from the pathoGen repo folder, venv active):
    python src/refilter.py <in_pipeline.csv> <out_pipeline.csv>
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from safety import check  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    df = pd.read_csv(src)
    for col in ["raw_text", "safe_text", "flag_codes"]:
        df[col] = df[col].fillna("")

    filtered, changed = 0, 0
    for i, row in df.iterrows():
        if bool(row["flagged"]) or row["raw_text"] == "":
            continue  # flagged for review or no evidence: nothing to filter
        res = check(row["raw_text"], row["pred"])
        codes = ",".join(sorted({c for c, _, _ in res["flags"]}))
        filtered += 1
        if res["safe_text"] != row["safe_text"]:
            changed += 1
        df.at[i, "safe_text"] = res["safe_text"]
        df.at[i, "unsafe_raw"] = bool(res["unsafe"])
        df.at[i, "fallback"] = bool(res["fallback"])
        df.at[i, "flag_codes"] = codes

    df.to_csv(dst, index=False)
    print(f"Re-filtered {filtered} generated rows; {changed} final texts changed -> {dst}")


if __name__ == "__main__":
    main()
