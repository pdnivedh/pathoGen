"""Copy the PathoGen result files into the repo's results/ folder.

The pipeline CSVs contain an "evidence" column (excerpts of doctor responses from the
OpenMed/MedDialog dataset). To avoid redistributing dataset text in a public repo,
that column is dropped from the copies. Model outputs, flags and metrics are kept.

Edit SRC and DST below if your folders are different, then run:
    python copy_results.py
"""
import shutil
from pathlib import Path

import pandas as pd

SRC = Path(r"E:\manoj\capstone_zipfiles\pathogen_part_4")
DST = Path(r"E:\manoj\genai_captsone\results")

CSV_FILES = [
    "pipeline_test_final.csv",
    "pipeline_test_final.csv.scored.csv",
    "pipeline_val100.csv",
    "pipeline_val100_r2.csv",
]
COPY_AS_IS = [
    "pipeline_test_final.csv.metrics.json",
]
DROP_COLUMNS = ["evidence", "evidence_short"]

DST.mkdir(parents=True, exist_ok=True)

for name in CSV_FILES:
    path = SRC / name
    if not path.exists():
        print("MISSING (skipped):", path)
        continue
    df = pd.read_csv(path)
    dropped = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    df.to_csv(DST / name, index=False)
    print(f"copied {name}  ({len(df)} rows, dropped columns: {dropped or 'none'})")

for name in COPY_AS_IS:
    path = SRC / name
    if not path.exists():
        print("MISSING (skipped):", path)
        continue
    shutil.copy(path, DST / name)
    print("copied", name)

print("\nDone. Files are in", DST)
