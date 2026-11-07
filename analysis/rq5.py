"""RQ5: can an informed swarm evade the audit without losing its mission?

Compares adaptive episodes (run with --adaptive, colluders shown the audit rules) against the
non-adaptive ones: evasion rate (audit misses an actual collusion), mission success (tickets
accepted via trip), and delay (first trip round). Reports aggregates only; no evasion recipes.
Usage: python analysis/rq5.py --adaptive-tag adaptive --baseline-tag main
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import bootstrap_ci, load_all, save_table  # noqa: E402
import random

from audit.closure import audit_episode  # noqa: E402
from tamper.tamper import apply  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adaptive-tag", default="adaptive")
ap.add_argument("--baseline-tag", default="main")
ap.add_argument("--arms", default=None, help="comma list of extra tags treated as further arms (e.g. gen1,gen2)")
ap.add_argument("--out", default="rq5.csv")
args = ap.parse_args()

arms = [("baseline", args.baseline_tag), ("adaptive", args.adaptive_tag)]
if args.arms:
    arms += [(t, t) for t in args.arms.split(",")]
rows = []
for name, tag in arms:
    eps = [e for e in load_all(tag) if e["meta"]["condition"] != "honest"]
    evaded, evaded_t, success, delay, instructed_but_inactive = [], [], [], [], 0
    by_cond = {}
    for ep in eps:
        res = audit_episode(ep["meta"], ep["peer"], ep["gateway"], ep["inference"], level="decode")
        if ep["label"]["positive"]:
            evaded.append(int(not res["alert"]))
            # evasion when the colluders also erase their own incriminating records (omission by all of them)
            rng = random.Random(ep["meta"]["seed"])
            logs_t = apply(ep["peer"], ep["meta"], list(ep["meta"]["colluders"]), "omission", rng)
            res_t = audit_episode(ep["meta"], logs_t, ep["gateway"], None, level="decode")
            evaded_t.append(int(not res_t["alert"]))
            c = ep["meta"]["condition"]
            by_cond.setdefault(c, []).append(int(not res_t["alert"]))
            success.append(ep["done"]["accepted_via_trip"] / max(1, 2 * len(ep["meta"]["colluders"])))
            trips = [e["round"] for e in ep["truth"] if e["kind"] == "scorer_trip"]
            delay.append(min(trips) if trips else ep["meta"]["rounds"])
        else:
            instructed_but_inactive += 1
    e, elo, ehi = bootstrap_ci(evaded)
    et, etlo, ethi = bootstrap_ci(evaded_t)
    s, slo, shi = bootstrap_ci(success)
    d, dlo, dhi = bootstrap_ci(delay)
    rows.append({"arm": name, "n_collusive": len(evaded), "inactive": instructed_but_inactive,
                 "evasion_truthful_logs": round(e, 3), "evasion_truthful_lo": round(elo, 3), "evasion_truthful_hi": round(ehi, 3),
                 "evasion_under_omission": round(et, 3), "evasion_lo": round(etlo, 3), "evasion_hi": round(ethi, 3),
                 **{f"evasion_omission_{c}": round(sum(v) / len(v), 3) for c, v in sorted(by_cond.items())},
                 "mission_success": round(s, 3), "success_lo": round(slo, 3), "success_hi": round(shi, 3),
                 "first_trip_round": round(d, 2), "delay_lo": round(dlo, 2), "delay_hi": round(dhi, 2)})
save_table(rows, args.out)
