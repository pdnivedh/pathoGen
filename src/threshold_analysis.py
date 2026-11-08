import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
df["error"] = df["true"] != df["pred_mc"]
n, n_err = len(df), df["error"].sum()
print(f"n={n}, errors={n_err}\n")
print("thr   flagged  flag%  acc_on_kept  errors_caught")
for thr in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    flagged = df["conf_mc"] < thr
    kept = df[~flagged]
    acc = (~kept["error"]).mean() if len(kept) else float("nan")
    caught = (flagged & df["error"]).sum()
    print(f"{thr:.2f}  {flagged.sum():6d}  {flagged.mean()*100:5.1f}  {acc:11.4f}  {caught}/{n_err}")