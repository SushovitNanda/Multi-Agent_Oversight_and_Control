"""Scale sweep summary: does closure survive as the swarm grows?

For each size tag (main = 12 agents, scale24, scale48, scale96) and each collusive episode, this
applies omission and spoof tampering with k liars chosen as (a) the colluders only up to a
fraction p of the swarm, for p in {7%, 33%} (random choice), and (b) hub-targeted liars of the
same size; records detection on truthful positives, false alarms on truthful negatives, the
erasable share versus the theory curve for that n, and the per-episode cost.
Usage: python analysis/scale.py --tags main,scale24,scale48,scale96
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

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="main,scale24,scale48,scale96")
ap.add_argument("--fractions", default="0.07,0.33")
args = ap.parse_args()

rows = []
for tag in args.tags.split(","):
    eps = load_all(tag)
    if not eps:
        print("no episodes for", tag)
        continue
    n = len(eps[0]["meta"]["agents"])
    for frac in (float(x) for x in args.fractions.split(",")):
        k = max(1, round(frac * n))
        for kind in ("omission", "spoof"):
            for selection in ("random", "targeted"):
                det, fp, eras, cost = [], [], defaultdict(list), []
                for i, ep in enumerate(eps):
                    meta = ep["meta"]
                    rng = random.Random(7919 * k + i)
                    liars = choose_liars(meta, k, rng, targeted=(selection == "targeted"))
                    # erasure is a property of the witness structure, so it is measured for a liar set
                    # chosen without regard to who colluded: uniformly random, or hubs first
                    if selection == "targeted" and meta.get("hubs"):
                        pool = [a for a in meta["agents"] if a not in meta["hubs"]]
                        rng.shuffle(pool)
                        liars_eras = (list(meta["hubs"]) + pool)[:k]
                    else:
                        liars_eras = rng.sample(list(meta["agents"]), k)
                    gw = ep["gateway"]
                    cost.append(ep["done"].get("credits") or 0)
                    if ep["label"]["positive"]:
                        logs = apply(ep["peer"], meta, liars, kind, rng)
                        det.append(int(audit_episode(meta, logs, gw, None, level="decode")["alert"]))
                        if kind == "omission":
                            eras[meta["topology"]].append(erasable_share(meta, ep["truth"], liars_eras))
                    else:
                        fp.append(int(audit_episode(meta, ep["peer"], gw, None, level="decode")["alert"]))
                m, lo, hi = bootstrap_ci(det)
                f, flo, fhi = bootstrap_ci(fp)
                row = {"tag": tag, "n_agents": n, "liar_fraction": frac, "k": k, "tamper": kind, "selection": selection,
                       "n_pos": len(det), "recall": round(m, 3), "recall_lo": round(lo, 3), "recall_hi": round(hi, 3),
                       "n_neg": len(fp), "fpr": round(f, 3), "fpr_hi": round(fhi, 3),
                       "credits_per_episode": round(sum(cost) / max(1, len(cost)), 4)}
                if kind == "omission":
                    for topo, vals in sorted(eras.items()):
                        em, elo, ehi = bootstrap_ci(vals)
                        theory = tree_worst_share(n, k) if selection == "targeted" and topo != "mesh" else random_share(n, k)
                        row[f"erasable_{topo}"] = round(em, 3)
                        row[f"erasable_{topo}_hi"] = round(ehi, 3)
                        row[f"theory_{topo}"] = round(theory, 3)
                rows.append(row)
                print({k2: v for k2, v in row.items() if not k2.endswith("_hi")}, flush=True)
save_table(rows, "scale.csv")
