"""Step 7 gate: per-episode summary of completion, collusion, audit violations and cost.
Usage: python scripts/pilot_report.py --tag pilot
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import load_all, save_table  # noqa: E402
from audit.closure import audit_episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="pilot")
ap.add_argument("--show-messages", type=int, default=0, help="print this many messages from collusive runs")
args = ap.parse_args()

rows = []
for ep in load_all(args.tag):
    m, d = ep["meta"], ep["done"]
    res = audit_episode(m, ep["peer"], ep["gateway"], ep["inference"], level="decode")
    rows.append({"episode": m["episode"], "topology": m["topology"], "condition": m["condition"],
                 "accepted": f"{d['accepted']}/{d['tickets']}", "via_trip": d["accepted_via_trip"],
                 "trips": d["scorer_trips"], "hidden": ep["label"]["hidden_transfers"], "positive": ep["label"]["positive"],
                 "msgs": d["messages"], "offlog_w/r": f"{d['offlog_writes']}/{d['offlog_reads']}",
                 "viol_msg": len(res["message"]), "viol_state": len(res["state"]), "viol_effect": len(res["effect"] or []),
                 "viol_know": len(res["knowledge"]), "alert": res["alert"],
                 "in_tok": d["input_tokens"], "cached": d["cached_tokens"], "out_tok": d["output_tokens"],
                 "credits": d["credits"], "api_cost": d.get("api_cost"), "secs": d["elapsed_s"]})
    if args.show_messages and m["condition"] != "honest":
        print(f"\n--- {m['episode']} colluders={m['colluders']}")
        n = 0
        for e in ep["truth"]:
            if e["kind"] in ("send", "offlog_write") and n < args.show_messages:
                n += 1
                who = f"{e['sender']}->{e['receiver']}" if e["kind"] == "send" else f"{e['actor']} cache_mkdir"
                print(f"  r{e['round']} {who}: {(e.get('content') or e.get('name'))[:160]}")
        if res["knowledge"]:
            print("  knowledge violations:", res["knowledge"][:6])
df = save_table(rows, f"pilot_report_{args.tag}.csv")
print("\ntotal credits (list):", round(df["credits"].sum(), 4), " api_cost:", round(df["api_cost"].fillna(0).sum(), 4))
