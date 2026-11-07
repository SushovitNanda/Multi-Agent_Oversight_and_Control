"""Lead figure: detection (and erasable share) against number of liars, by topology, with theory overlay.
Usage: python analysis/figures.py --tag main
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.theory import random_share, tree_worst_share  # noqa: E402
from config import RESULTS_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
args = ap.parse_args()

PALETTE = {"mesh": "#4C78A8", "star": "#F58518", "lanes": "#54A24B", "theory": "#9D9D9D"}

er = pd.read_csv(RESULTS_DIR / f"rq2_erasure_{args.tag}.csv")
fig, ax = plt.subplots(figsize=(7, 4.2))
ks = list(range(0, 13))
ax.plot(ks, [100 * random_share(12, k) for k in ks], color=PALETTE["theory"], ls="--", label="Prop. 1: random, any graph")
ax.plot(ks, [100 * tree_worst_share(12, k) for k in ks], color=PALETTE["theory"], ls=":", label="Prop. 2: targeted, any tree")
for topo, g in er.groupby("topology"):
    base = topo.replace("_targeted", "")
    ax.errorbar(g["k"], 100 * g["measured_erasable"], yerr=[100 * (g["measured_erasable"] - g["lo"]), 100 * (g["hi"] - g["measured_erasable"])],
                marker="o" if "targeted" not in topo else "s", ls="-" if "targeted" not in topo else "--",
                color=PALETTE.get(base, "#333"), capsize=2, label=f"measured {topo}")
ax.set_xlabel("Colluding agents k of 12")
ax.set_ylabel("Messages erasable without contradiction (%)")
ax.set_title("Undetected erasure vs. witness theory")
ax.grid(alpha=0.3)
ax.legend(fontsize=7)
fig.tight_layout()
out = RESULTS_DIR / f"fig_erasure_{args.tag}.png"
fig.savefig(out, dpi=160)
print("->", out)

rq2 = pd.read_csv(RESULTS_DIR / f"rq2_{args.tag}.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, sel in zip(axes, ("random", "targeted")):
    for kind, g in rq2[rq2.selection == sel].groupby("tamper"):
        ax.errorbar(g["k"], g["recall"], yerr=[g["recall"] - g["recall_lo"], g["recall_hi"] - g["recall"]],
                    marker="o", capsize=2, label=kind)
    ax.set_title(f"Closure audit recall, {sel} liars")
    ax.set_xlabel("k liars")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Recall (episode level)")
axes[1].legend(fontsize=7)
fig.tight_layout()
out = RESULTS_DIR / f"fig_recall_{args.tag}.png"
fig.savefig(out, dpi=160)
print("->", out)
