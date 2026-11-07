"""RQ4: which vantage point preserves detection with the least retained data?

Ablates the views (peer logs, inference gateway, tool gateway), content level (full, 10% sample,
metadata only) and a cross-organisation split (two gateways that do not share records).
Usage: python analysis/rq4.py --tag main
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import bootstrap_ci, load_all, save_table  # noqa: E402
from audit.closure import audit_episode  # noqa: E402
from tamper.tamper import apply, choose_liars  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--tamper", default="omission", help="tamper class applied to positive episodes")
args = ap.parse_args()
eps = load_all(args.tag)


def _keep(e, rng):
    import hashlib
    key = e.get("msg_id") or e.get("call_id") or f"{e.get('actor')}-{e.get('round')}"
    h = int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 100
    return h < 10


def strip_content(events, mode, rng):
    out = []
    for e in events:
        e = dict(e)
        if mode == "metadata":
            e.pop("content", None)
            if "arguments" in e:
                e["arguments"] = "{}"
            e.pop("output_text", None)
            e["tool_calls"] = []
        elif mode == "sample10" and not _keep(e, rng):
            e.pop("content", None)
            if "arguments" in e:
                e["arguments"] = "{}"
            e.pop("output_text", None)
            e["tool_calls"] = []
        out.append(e)
    return out


def bytes_of(obj):
    import json
    return len(json.dumps(obj))


CONFIGS = [
    ("peer_only", {"peer"}), ("gateway_only", {"gateway"}), ("inference_only", {"inference"}),
    ("peer+gateway", {"peer", "gateway"}), ("peer+inference", {"peer", "inference"}), ("all", {"peer", "gateway", "inference"}),
]
rows = []
for name, views in CONFIGS:
    for content in ("full", "sample10", "metadata"):
        det, fp, size = [], [], []
        for i, ep in enumerate(eps):
            rng = random.Random(i)
            liars = choose_liars(ep["meta"], args.k, rng)
            tampered = apply(ep["peer"], ep["meta"], liars, args.tamper, rng) if ep["label"]["positive"] else ep["peer"]
            peer = {a: strip_content(v, content, rng) for a, v in tampered.items()} if "peer" in views else {a: [] for a in tampered}
            gw = strip_content(ep["gateway"], content, rng) if "gateway" in views else None
            inf = strip_content(ep["inference"], content, rng) if "inference" in views else None
            if "peer" not in views:
                # Without peer logs, the only closure left is knowledge (via gateway/inference) and effect is undefined.
                gw_for_effect = None
            else:
                gw_for_effect = gw
            res = audit_episode(ep["meta"], peer, gw_for_effect, inf, level="decode")
            if "peer" not in views and gw is not None:
                from audit.closure import knowledge_closure, sightings_from_views
                s = sightings_from_views(ep["meta"]["owners"], None, gw, inf, level="decode")
                res["knowledge"] = knowledge_closure(ep["meta"]["owners"], s, [])
                res["alert"] = bool(res["knowledge"])
            size.append(bytes_of(peer) + (bytes_of(gw) if gw else 0) + (bytes_of(inf) if inf else 0))
            (det if ep["label"]["positive"] else fp).append(int(res["alert"]))
        m, lo, hi = bootstrap_ci(det)
        f, _, fhi = bootstrap_ci(fp)
        rows.append({"views": name, "content": content, "n_pos": len(det), "recall": round(m, 3), "lo": round(lo, 3),
                     "hi": round(hi, 3), "n_neg": len(fp), "fpr": round(f, 3), "fpr_hi": round(fhi, 3),
                     "kb_per_episode": round(sum(size) / max(1, len(size)) / 1024, 1)})
save_table(rows, f"rq4_{args.tag}.csv" if args.tamper == "omission" else f"rq4_{args.tag}_{args.tamper}.csv")
