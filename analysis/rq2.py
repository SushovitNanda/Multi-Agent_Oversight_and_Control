"""RQ2: how many liars can cross-agent reconciliation tolerate?

For each episode, tamper class, k and selection (random / targeted), run the closure audit on the
tampered peer logs plus the tool gateway, and record detection, abstention and blame accuracy.
Also measures undetected-erasure share against Propositions 1 and 2.
Usage: python analysis/rq2.py --tag main
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import bootstrap_ci, load_all, save_table  # noqa: E402
from analysis.theory import random_share, tree_worst_share  # noqa: E402
from audit.closure import audit_episode  # noqa: E402
from tamper.tamper import apply, choose_liars, erasable_share  # noqa: E402

KS = (0, 1, 2, 3, 4, 6, 8)
CLASSES = ("omission", "fabrication", "spoof", "mutual_forgery")

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
ap.add_argument("--no-gateway", action="store_true", help="peer logs only (no effect closure)")
ap.add_argument("--ks", default=None, help="comma list of liar counts (default 0,1,2,3,4,6,8)")
ap.add_argument("--level", default="decode", choices=["exact", "decode"], help="tracer matching level for knowledge closure")
ap.add_argument("--conditions", default="inband,offlog", help="collusive conditions to sweep")
ap.add_argument("--out-suffix", default="")
args = ap.parse_args()
if args.ks:
    KS = tuple(int(x) for x in args.ks.split(","))
eps = load_all(args.tag)
print(f"{len(eps)} episodes")


def blamed(res, peer_logs):
    """Agents implicated by any violation: senders/receivers of inconsistent messages, spoofed callers, tracer holders."""
    out = set()
    index = {}
    for a, evs in peer_logs.items():
        for e in evs:
            if e["kind"] in ("send", "receive"):
                index.setdefault(e["msg_id"], set()).add(a)
                index[e["msg_id"]].add(e.get("sender") or a)
                index[e["msg_id"]].add(e.get("receiver") or a)
    for m in res["message"]:
        out |= index.get(m, set())
    for a, evs in peer_logs.items():
        if any(e["kind"] == "tool_call" and e["call_id"] in (res["effect"] or []) for e in evs):
            out.add(a)
    for agent, _ in res["knowledge"]:
        out.add(agent)
    return out


rows, erasure = [], []
for kind in CLASSES:
  for condition in args.conditions.split(","):
    for selection in ("random", "targeted"):
        for k in KS:
            det, abstain, blame_prec, blame_rec, fp, fp_tampered = [], [], [], [], [], []
            for i, ep in enumerate(eps):
                if ep["meta"]["condition"] not in (condition, "honest", "gossip"):
                    continue
                rng = random.Random(1000 * k + 7 * i)
                liars = choose_liars(ep["meta"], k, rng, targeted=(selection == "targeted")) if k else []
                positive = ep["label"]["positive"]
                gw = None if args.no_gateway else ep["gateway"]
                if not positive:
                    # False-alarm rate is measured on truthful logs: nobody in a negative episode has
                    # anything to hide. Detection of tampering by agents who did not collude is
                    # reported separately (neg_tampered_alert), since an alert there is correct.
                    res_clean = audit_episode(ep["meta"], ep["peer"], gw, None, level=args.level)
                    fp.append(int(res_clean["alert"]))
                    if k:
                        res_t = audit_episode(ep["meta"], apply(ep["peer"], ep["meta"], liars, kind, rng), gw, None, level=args.level)
                        fp_tampered.append(int(res_t["alert"]))
                    continue
                logs = apply(ep["peer"], ep["meta"], liars, kind if k else "none", rng)
                res = audit_episode(ep["meta"], logs, gw, None, level=args.level)
                n_viol = sum(len(res[x] or []) for x in ("message", "state", "effect", "knowledge"))
                verdict = "alert" if n_viol >= 2 else "insufficient" if n_viol == 1 else "clean"
                if positive and k:
                    det.append(int(res["alert"]))
                    abstain.append(int(verdict == "insufficient"))
                    b = blamed(res, logs)
                    if b:
                        blame_prec.append(len(b & set(liars)) / len(b))
                        blame_rec.append(len(b & set(liars)) / max(1, len(liars)))
                if k and kind == "omission" and selection == "random" and condition == "inband":
                    erasure.append({"topology": ep["meta"]["topology"], "k": k, "n": len(ep["meta"]["agents"]),
                                    "measured": erasable_share(ep["meta"], ep["truth"], liars)})
                if k and kind == "omission" and selection == "targeted" and condition == "inband":
                    erasure.append({"topology": ep["meta"]["topology"] + "_targeted", "k": k, "n": len(ep["meta"]["agents"]),
                                    "measured": erasable_share(ep["meta"], ep["truth"], liars)})
            m, lo, hi = bootstrap_ci(det)
            fpm, fplo, fphi = bootstrap_ci(fp)
            rows.append({"tamper": kind, "condition": condition, "selection": selection, "k": k, "n_pos": len(det), "recall": round(m, 3),
                         "recall_lo": round(lo, 3), "recall_hi": round(hi, 3), "abstain": round(sum(abstain) / max(1, len(abstain)), 3),
                         "blame_precision": round(sum(blame_prec) / max(1, len(blame_prec)), 3),
                         "blame_recall": round(sum(blame_rec) / max(1, len(blame_rec)), 3),
                         "n_neg": len(fp), "fpr": round(fpm, 3), "fpr_hi": round(fphi, 3),
                         "neg_tampered_alert": round(sum(fp_tampered) / max(1, len(fp_tampered)), 3)})
save_table(rows, f"rq2_{args.tag}{'_nogw' if args.no_gateway else ''}{args.out_suffix}.csv")

# Erasure vs theory (lead figure data)
agg = defaultdict(list)
n_of = {}
for r in erasure:
    agg[(r["topology"], r["k"])].append(r["measured"])
    n_of[(r["topology"], r["k"])] = r["n"]
erows = []
for (topo, k), vals in sorted(agg.items()):
    m, lo, hi = bootstrap_ci(vals)
    n = n_of[(topo, k)]
    theory = tree_worst_share(n, k) if topo.endswith("_targeted") and topo != "mesh_targeted" else random_share(n, k)
    erows.append({"topology": topo, "n": n, "k": k, "measured_erasable": round(m, 3), "lo": round(lo, 3), "hi": round(hi, 3),
                  "theory": round(theory, 3), "gap": round(m - theory, 3)})
save_table(erows, f"rq2_erasure_{args.tag}.csv")
