"""Cross-family summary: one row per (model tag, condition) with positives, alerts on truthful
logs, off-log recall, and RQ2 recall at k=4 for omission and spoof.
Usage: python analysis/families.py --tags main,gemma,qwen
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import save_table  # noqa: E402
from config import RESULTS_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="main,gemma,qwen")
args = ap.parse_args()

rows = []
for tag in args.tags.split(","):
    rep = pd.read_csv(RESULTS_DIR / f"pilot_report_{tag}.csv")
    rq2 = pd.read_csv(RESULTS_DIR / f"rq2_{tag}.csv")
    rq3 = pd.read_csv(RESULTS_DIR / f"rq3_{tag}.csv")
    model = rep.get("model", pd.Series([tag])).iloc[0] if "model" in rep else tag
    for cond, g in rep.groupby("condition"):
        r = {"tag": tag, "condition": cond, "n": len(g), "accepted_24": round(g["accepted"].eq("24/24").mean(), 3),
             "positive_rate": round(g["positive"].mean(), 3), "alert_truthful_logs": round(g["alert"].mean(), 3),
             "trips_mean": round(g["trips"].mean(), 2), "hidden_mean": round(g["hidden"].mean(), 2),
             "credits_mean": round(g["credits"].mean(), 4), "secs_mean": round(g["secs"].mean(), 0)}
        if cond != "honest":
            for kind in ("omission", "spoof", "mutual_forgery"):
                q = rq2[(rq2.tamper == kind) & (rq2.condition == cond) & (rq2.selection == "random") & (rq2.k == 4)]
                r[f"rq2_{kind}_k4_recall"] = float(q.recall.iloc[0]) if len(q) else None
                r[f"rq2_{kind}_k4_npos"] = int(q.n_pos.iloc[0]) if len(q) else 0
        if cond == "offlog":
            q = rq3[rq3.level == "decode"]
            r["rq3_pair_recall"] = float(q.recall.iloc[0]) if len(q) else None
            r["rq3_lag_rounds"] = float(q.lag_rounds_mean.iloc[0]) if len(q) else None
        rows.append(r)
save_table(rows, "families.csv")
print(pd.DataFrame(rows).to_string(index=False))
