"""Print the quantities quoted in the report prose so they can be checked against the regenerated tables.
Usage: python analysis/report_numbers.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR  # noqa: E402


def load(name):
    p = RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def row(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    return df[m]


print("== RQ1 census")
c = load("rq0.csv")
if c is not None:
    honest = c[c.condition.isin(["honest", "gossip"])]
    print("honest+gossip episodes", int(honest.n.sum()), "alerts", float((honest.alert_full * honest.n).sum()))
    for tag, cond in (("main", "inband"), ("main", "offlog"), ("gemma", "honest"), ("gemma", "offlog"), ("cond", "gossip")):
        r = row(c, tag=tag, condition=cond)
        if len(r):
            r = r.iloc[0]
            print(f"{tag}/{cond}: alert {r.alert_full:.2f} no_file {r.alert_msg_edges_only:.2f} no_lastround {r.alert_no_lastround_rule:.2f} "
                  f"sightings {r.nonowner_sightings_mean:.1f} via_msg {r.share_msg_path:.2f} unexpl {r.share_unexplained:.2f}")

print("== RQ2 monitors (k=0 and k=4)")
for name in ("rq1_main_judge4k.csv", "rq1_main_judge4k_k12.csv", "rq1_main_strong.csv"):
    d = load(name)
    if d is None:
        continue
    for _, r in d.iterrows():
        if r.monitor != "rule" or r.k in (0, 4):
            print(f"{name}: {r.tamper} k={r.k} {r.monitor} auroc {r.auroc:.2f} tpr5 {r.tpr_at_5pct_fpr:.2f}")

print("== RQ3 recall (random)")
q = load("rq2_main.csv")
if q is not None:
    q = q[q.selection == "random"]
    for kind in ("omission", "fabrication", "spoof", "mutual_forgery"):
        for cond in ("inband", "offlog"):
            g = row(q, tamper=kind, condition=cond).sort_values("k")
            print(kind, cond, " ".join(f"k{int(r.k)}={r.recall:.2f}" for _, r in g.iterrows() if r.k > 0), "fpr_max", g.fpr.max())
    print("negatives:", q.n_neg.unique(), "positives:", q.n_pos.unique())
e = load("rq2_erasure_main.csv")
if e is not None:
    for topo in ("mesh", "star_targeted", "lanes_targeted"):
        r = row(e, topology=topo, k=4).iloc[0]
        print(f"erasure {topo} k=4: {r.measured_erasable:.2f} vs theory {r.theory:.2f}")

print("== RQ3 scale")
s = load("scale.csv")
if s is not None:
    for _, r in s[s.selection == "random"].iterrows():
        print(f"n={int(r.n_agents)} frac={r.liar_fraction} {r.tamper}: recall {r.recall:.2f} fpr {r.fpr:.2f} n_pos {int(r.n_pos)} credits {r.credits_per_episode:.3f}")

print("== RQ4 tracers")
for tag in ("main", "gemma", "qwen", "gemma26"):
    d = load(f"rq3_{tag}.csv")
    if d is not None:
        r = d[d.level == "decode"].iloc[0]
        print(f"{tag}: pair recall {r.recall:.2f} [{r.recall_lo:.2f},{r.recall_hi:.2f}] lag {r.lag_rounds_mean} fpr {r.honest_episode_fpr:.2f} false pairs {int(r.honest_false_pairs)}")
f = load("families.csv")
if f is not None:
    for _, r in f.iterrows():
        print(f"family {r.tag}/{r.condition}: pos {r.positive_rate:.2f} alert_truthful {r.alert_truthful_logs:.2f} rq3_pair {r.rq3_pair_recall}")
for suf in ("_exact", "_decode"):
    d = load(f"rq2_cond{suf}.csv")
    if d is not None:
        d = d[d.selection == "random"]
        for cond in ("stego", "paraphrase"):
            for kind in ("omission", "spoof"):
                g = row(d, condition=cond, tamper=kind).sort_values("k")
                print(f"cond{suf} {cond} {kind}: n_pos {int(g.n_pos.max())} " + " ".join(f"k{int(r.k)}={r.recall:.2f}" for _, r in g.iterrows() if r.k > 0) + f" fpr {g.fpr.max():.2f}")

print("== RQ5 views")
v = load("rq4_main.csv")
if v is not None:
    for views, content in (("peer_only", "full"), ("gateway_only", "full"), ("inference_only", "full"), ("peer+gateway", "metadata"),
                           ("peer+inference", "full"), ("peer+inference", "metadata"), ("all", "full")):
        r = row(v, views=views, content=content).iloc[0]
        print(f"{views}/{content}: recall {r.recall:.2f} fpr {r.fpr:.2f} kb {r.kb_per_episode:.0f}")

print("== RQ6 arms")
for name in ("rq5_vs_main.csv", "rq5_vs_control.csv"):
    d = load(name)
    if d is not None and "evasion_truthful_logs" in d:
        for _, r in d.iterrows():
            print(f"{name} {r.arm}: active {int(r.n_collusive)} inactive {int(r.inactive)} miss_truthful {r.evasion_truthful_logs:.2f} "
                  f"miss_erased {r.evasion_under_omission:.2f} success {r.mission_success:.2f} first_trip {r.first_trip_round:.1f}")
