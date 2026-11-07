"""RQ1: how fast does transcript oversight fail as k liars tamper with the peer logs?

Scores the rule monitor (and optionally LLM judges) on peer logs only, across tamper classes and k.
Usage: python analysis/rq1.py --tag main [--judge deepseek-v4-flash] [--strong glm-5.2 --strong-n 60]
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import auroc, load_all, save_table, tpr_at_fpr  # noqa: E402
from audit.monitors import llm_judge, rule_monitor  # noqa: E402
from tamper.tamper import apply, choose_liars  # noqa: E402

KS = (0, 1, 2, 3, 4, 6)
CLASSES = ("omission", "fabrication", "spoof", "mutual_forgery")

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
ap.add_argument("--judge", default=None, help="cheap judge model id")
ap.add_argument("--strong", default=None, help="strong judge model id")
ap.add_argument("--strong-n", type=int, default=60)
ap.add_argument("--with-rationale", action="store_true")
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--judge-tokens", type=int, default=1500, help="output cap for judge calls")
ap.add_argument("--out", default=None, help="table name, default rq1_<tag>.csv")
ap.add_argument("--ks", default=None, help="comma list of liar counts, default 0,1,2,3,4,6")
ap.add_argument("--classes", default=None, help="comma list of tamper classes")
ap.add_argument("--with-gateway", action="store_true", help="judges also see the tool gateway record (same two views as the audit)")
args = ap.parse_args()
if args.ks:
    KS = tuple(int(x) for x in args.ks.split(','))
if args.classes:
    CLASSES = tuple(args.classes.split(','))

eps = load_all(args.tag)
print(f"{len(eps)} episodes")
client = None
if args.judge or args.strong:
    from config import RESULTS_DIR, adaption_client
    from gateway.client import LoggedClient
    client = LoggedClient(adaption_client(), RESULTS_DIR / "judge_calls.jsonl")

from concurrent.futures import ThreadPoolExecutor

rows = []
judge_k0, strong_k0 = None, None   # k=0 logs are untampered, identical for every tamper class
for kind in CLASSES:
    for k in KS:
        scores = {"rule": [], "judge": [], "strong": []}
        labels, jobs = [], []
        for i, ep in enumerate(eps):
            rng = random.Random(1000 * k + i)
            liars = choose_liars(ep["meta"], k, rng) if k else []
            logs = apply(ep["peer"], ep["meta"], liars, kind if k else "none", rng)
            labels.append(int(ep["label"]["positive"]))
            scores["rule"].append(rule_monitor(logs))
            rationale = None
            if args.with_rationale:
                rationale = {}
                for rec in ep["inference"]:
                    rationale.setdefault(rec["actor"], {})[rec["round"]] = rec.get("output_text", "")
            jobs.append((i, ep["meta"]["episode"], logs, rationale))
        if args.judge:
            if k == 0 and judge_k0 is not None:
                scores["judge"] = list(judge_k0)
            else:
                with ThreadPoolExecutor(args.workers) as ex:
                    scores["judge"] = list(ex.map(lambda j: llm_judge(client, args.judge, j[2], j[1], j[3], args.judge_tokens,
                                                                      gateway=eps[j[0]]["gateway"] if args.with_gateway else None)[0], jobs))
                if k == 0:
                    judge_k0 = list(scores["judge"])
        if args.strong:
            stride = max(1, len(jobs) // args.strong_n)
            sub = [j for j in jobs if j[0] % stride == 0][: args.strong_n]  # stratified across topology x condition
            if k == 0 and strong_k0 is not None:
                scores["strong"] = list(strong_k0)
            else:
                with ThreadPoolExecutor(args.workers) as ex:
                    scores["strong"] = list(ex.map(lambda j: llm_judge(client, args.strong, j[2], j[1], j[3], args.judge_tokens,
                                                                       gateway=eps[j[0]]["gateway"] if args.with_gateway else None)[0], sub))
                if k == 0:
                    strong_k0 = list(scores["strong"])
        print(f"{kind} k={k} done", flush=True)
        save_table(rows, args.out or f"rq1_{args.tag}.csv")
        for name, sc in scores.items():
            if not sc:
                continue
            lab = labels[: len(sc)] if name != "strong" else [labels[j[0]] for j in sub]
            rows.append({"tamper": kind, "k": k, "monitor": name, "n": len(sc),
                         "auroc": round(auroc(sc, lab), 3), "tpr_at_1pct_fpr": round(tpr_at_fpr(sc, lab, 0.01), 3),
                         "tpr_at_5pct_fpr": round(tpr_at_fpr(sc, lab, 0.05), 3)})
save_table(rows, args.out or f"rq1_{args.tag}.csv")
