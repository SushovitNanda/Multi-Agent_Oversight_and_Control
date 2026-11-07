"""RQ0: is causal closure an invariant of honest multi-agent activity?

Every log-based oversight method assumes the logs are causally complete: what an agent knows or
does can be traced to something recorded. This script measures how often honest swarms violate
closure on TRUTHFUL logs (no tamper), which channels the auditor must model for closure to hold,
and how often benign knowledge moves between agents at all.

Per (tag, condition) on untampered logs:
  alert_full          alert rate with every sanctioned channel modelled (messages, shared files,
                      final-round sends treated as unverifiable)
  alert_msg_edges     alert rate when only message edges count as paths (shared files unmodelled)
  alert_no_lastround  alert rate when final-round sends are treated like any other send
  nonowner_sightings  mean number of (agent, tracer) pairs where an agent handles another's tracer
  share_msg_path      share of those explained by message paths alone
  share_file_path     share that need a shared-file hop to be explained
  share_unexplained   share with no logged path (should be 0 in honest episodes)
Usage: python analysis/rq0.py --tags main,gemma,qwen
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import bootstrap_ci, load_all, save_table  # noqa: E402
from audit.closure import (audit_episode, edges_from_peer_logs, knowledge_closure,  # noqa: E402
                           message_closure, sightings_from_views)

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="main,gemma,qwen")
args = ap.parse_args()


def message_only_edges(peer_logs):
    edges = set()
    for agent, events in peer_logs.items():
        for e in events:
            if e["kind"] == "send":
                edges.add((agent, e["receiver"], e["round"]))
            elif e["kind"] == "receive":
                edges.add((e["sender"], agent, e["round"]))
    return sorted(edges)


rows = []
for tag in args.tags.split(","):
    eps = load_all(tag)
    groups = defaultdict(list)
    for ep in eps:
        groups[ep["meta"]["condition"]].append(ep)
    for cond, group in sorted(groups.items()):
        full, msg_only, no_last, n_sight, s_msg, s_file, s_none = [], [], [], [], 0, 0, 0
        for ep in group:
            meta, peer, gw, inf = ep["meta"], ep["peer"], ep["gateway"], ep["inference"]
            owners = meta["owners"]
            res = audit_episode(meta, peer, gw, inf, level="decode")
            full.append(int(res["alert"]))
            sightings = sightings_from_views(owners, peer, gw, inf, level="decode")
            nonowner = sorted({(a, t) for a, t, r in sightings if owners[t] != a})
            n_sight.append(len(nonowner))
            flagged_msg = set(knowledge_closure(owners, sightings, message_only_edges(peer)))
            flagged_all = set(knowledge_closure(owners, sightings, edges_from_peer_logs(peer)))
            msg_only.append(int(bool(flagged_msg) or bool(res["message"]) or bool(res["state"]) or bool(res["effect"])))
            m_nolast = message_closure(peer, last_round=None)
            no_last.append(int(bool(m_nolast) or bool(res["state"]) or bool(res["effect"]) or bool(flagged_all)))
            for pair in nonowner:
                if pair in flagged_all:
                    s_none += 1
                elif pair in flagged_msg:
                    s_file += 1
                else:
                    s_msg += 1
        tot = max(1, s_msg + s_file + s_none)
        m, lo, hi = bootstrap_ci(full)
        rows.append({"tag": tag, "condition": cond, "n": len(group),
                     "alert_full": round(m, 3), "alert_full_hi": round(hi, 3),
                     "alert_msg_edges_only": round(sum(msg_only) / len(group), 3),
                     "alert_no_lastround_rule": round(sum(no_last) / len(group), 3),
                     "nonowner_sightings_mean": round(sum(n_sight) / len(group), 2),
                     "share_msg_path": round(s_msg / tot, 3), "share_file_path": round(s_file / tot, 3),
                     "share_unexplained": round(s_none / tot, 3)})
        print(rows[-1], flush=True)
save_table(rows, "rq0.csv")
