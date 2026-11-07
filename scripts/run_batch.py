"""Steps 7 and 8: run pilot / main episodes. Resumable: skips episodes whose done.json exists.

Examples:
  python scripts/run_batch.py --tag pilot --seeds 0-4 --topologies mesh,star,lanes --conditions honest,inband,offlog
  python scripts/run_batch.py --tag main --seeds 0-19
  python scripts/run_batch.py --tag dry --seeds 0-1 --scripted
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MAIN_MODEL, RUNS_DIR  # noqa: E402
from sim.episode import run_episode  # noqa: E402


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
ap.add_argument("--model", default=MAIN_MODEL)
ap.add_argument("--seeds", default="0-19")
ap.add_argument("--topologies", default="mesh,star,lanes")
ap.add_argument("--conditions", default="honest,inband,offlog")
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--agents", type=int, default=12, help="swarm size (12, 24, 48, 96)")
ap.add_argument("--parallel", type=int, default=3, help="episodes in parallel (each uses 12 threads)")
ap.add_argument("--agent-workers", type=int, default=12)
ap.add_argument("--scripted", action="store_true")
ap.add_argument("--targeted", action="store_true")
ap.add_argument("--adaptive", action="store_true")
ap.add_argument("--feedback-file", default=None, help="text file of aggregate audit feedback shown to adaptive colluders")
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()
FEEDBACK = open(args.feedback_file, encoding="utf-8").read() if args.feedback_file else None

jobs = []
for seed in parse_seeds(args.seeds):
    for topo in args.topologies.split(","):
        for cond in args.conditions.split(","):
            eid = f"{args.tag}_{topo}_{cond}_s{seed:02d}"
            if (RUNS_DIR / eid / "done.json").exists():
                continue
            jobs.append((eid, topo, cond, seed))
print(f"{len(jobs)} episodes to run with {args.model} -> {RUNS_DIR}")


def one(job):
    eid, topo, cond, seed = job
    t0 = time.time()
    try:
        s = run_episode(eid, topo, cond, seed, args.model, scripted=args.scripted, rounds=args.rounds,
                        targeted=args.targeted, adaptive=args.adaptive, verbose=args.verbose,
                        max_workers=min(args.agent_workers, args.agents), n_agents=args.agents, feedback=FEEDBACK)
        s["episode"] = eid
        return s
    except Exception as exc:  # noqa: BLE001
        return {"episode": eid, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "elapsed_s": time.time() - t0}


total_credits = 0.0
with ThreadPoolExecutor(max_workers=args.parallel) as pool:
    futures = [pool.submit(one, j) for j in jobs]
    for fut in as_completed(futures):
        s = fut.result()
        total_credits += s.get("credits", 0) or 0
        print(json.dumps(s))
print(f"done. credits spent (list price): {total_credits:.3f}")
