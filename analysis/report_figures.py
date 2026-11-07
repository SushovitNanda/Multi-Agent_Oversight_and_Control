"""Report figures and inline table rows, built from the CSV tables in results/.

Figures go to Report/figures/ (the flow diagram is a TikZ source in that folder, rendered to PNG
with pdflatex + pdftoppm). Table rows are written straight into Report/main.tex between marker
comments:   % BEGIN ROWS <name>  ...  % END ROWS <name>
and single values into lines tagged   % VALUE <name>
so the report carries its own table code and still refreshes from regenerated CSVs.
Usage: python analysis/report_figures.py
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.theory import random_share, tree_worst_share  # noqa: E402
from config import RESULTS_DIR  # noqa: E402

REPORT = Path(__file__).resolve().parents[1] / "Report"
FIG = REPORT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MAIN = REPORT / "main.tex"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7.5, "font.family": "sans-serif"})
C = {"audit": "#2A6F97", "rule": "#9D9D9D", "judge": "#F58518", "strong": "#B5451B", "mesh": "#4C78A8", "star": "#F58518",
     "lanes": "#54A24B", "theory": "#9D9D9D", "omission": "#4C78A8", "fabrication": "#54A24B", "spoof": "#F58518",
     "mutual_forgery": "#B279A2"}
NICE = {"omission": "omission", "fabrication": "fabrication", "spoof": "spoof", "mutual_forgery": "mutual forgery"}
MODEL = {"main": "deepseek-v4-flash", "gemma": "gemma-4-31b-it", "qwen": "qwen3.8-27b", "gemma26": "gemma-4-26b-a4b-it",
         "cond": "deepseek-v4-flash", "scale24": "deepseek-v4-flash, 24 agents", "scale48": "deepseek-v4-flash, 48 agents",
         "scale96": "deepseek-v4-flash, 96 agents", "adaptive_ds": "deepseek-v4-flash, adaptive", "control": "qwen3.5-397b, control",
         "adaptive": "qwen3.5-397b, adaptive", "gen1": "deepseek-v4-flash, gen 1", "gen2": "deepseek-v4-flash, gen 2",
         "long40": "deepseek-v4-flash, 40 rounds", "stegooff": "deepseek-v4-flash, encoded cache"}
ROWS, VALUES = {}, {}


def load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        print("missing", name)
        return None
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        print("empty", name)
        return None


def rows(name, lines):
    ROWS[name] = "\n".join(lines) + "\n"


def fmt(x, nd=2):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    print("->", FIG / name)


# ---------------------------------------------------------------- flow diagram (TikZ -> PDF -> PNG)
def render_flow():
    src = FIG / "fig_flow.tex"
    if not src.exists():
        print("no fig_flow.tex")
        return
    if shutil.which("pdflatex") is None or shutil.which("pdftoppm") is None:
        print("pdflatex or pdftoppm not found; fig_flow.png not rendered")
        return
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(src, Path(td) / "fig_flow.tex")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig_flow.tex"], cwd=td, capture_output=True, text=True)
        if r.returncode != 0:
            print("flow diagram failed:", r.stdout[-800:])
            return
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", "fig_flow.pdf", str(FIG / "fig_flow")], cwd=td, check=True)
    print("->", FIG / "fig_flow.png")


render_flow()

# ---------------------------------------------------------------- RQ2: monitors vs audit
j4 = load("rq1_main_judge4k.csv")
j12 = load("rq1_main_judge4k_k12.csv")
st = load("rq1_main_strong.csv")
rq2 = load("rq2_main.csv")
if j4 is not None and rq2 is not None:
    parts = [j4]
    if j12 is not None:
        parts.append(j12[j12.k.isin([1, 2])])
    jd = pd.concat(parts).drop_duplicates(["tamper", "k", "monitor"], keep="first")
    if st is not None:
        jd = pd.concat([jd, st[st.monitor == "strong"]])
    r2 = rq2[rq2.selection == "random"]

    def audit_pooled(kind, k):
        g = r2[(r2.tamper == kind) & (r2.k == k)]
        n = g.n_pos.sum()
        return (g.recall * g.n_pos).sum() / n if n else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4), sharey=True)
    for ax, kind in zip(axes, ("omission", "spoof")):
        for mon, lab, col, mk in (("rule", "rule monitor", C["rule"], "s"), ("judge", "LLM judge (deepseek-v4-flash)", C["judge"], "o"),
                                  ("strong", "strong judge (qwen3.5-397b, n=60)", C["strong"], "^")):
            g = jd[(jd.tamper == kind) & (jd.monitor == mon)].sort_values("k")
            if len(g):
                ax.plot(g.k, g.tpr_at_5pct_fpr, marker=mk, color=col, label=f"{lab}: TPR at 5% FPR")
        ks = [k for k in (1, 2, 3, 4, 6, 8) if not np.isnan(audit_pooled(kind, k))]
        ax.plot(ks, [audit_pooled(kind, k) for k in ks], marker="D", color=C["audit"], lw=2, label="closure audit: recall at 0% FPR")
        ax.set_title(NICE[kind])
        ax.set_xlabel("liars k (of 12)")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("detection rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    savefig(fig, "fig_monitors.png")

    lines = []
    for kind in ("omission", "fabrication", "spoof", "mutual_forgery"):
        for k in sorted(jd[jd.tamper == kind].k.unique()):
            g = jd[(jd.tamper == kind) & (jd.k == k)].set_index("monitor")
            a = audit_pooled(kind, k)
            cells = [NICE[kind], str(int(k))]
            for mon in ("rule", "judge", "strong"):
                cells.append(f"{g.loc[mon, 'auroc']:.2f} / {g.loc[mon, 'tpr_at_5pct_fpr']:.2f}" if mon in g.index else "--")
            cells.append("--" if np.isnan(a) else f"{a:.2f} / 0.00")
            lines.append(" & ".join(cells) + r" \\")
    rows("tab_monitors", lines)
    # compact main-body table: untampered and four liars
    lines = []
    for kind, k, name in (("omission", 0, "untampered"), ("omission", 4, "omission"), ("spoof", 4, "spoof"), ("mutual_forgery", 4, "mutual forgery")):
        g = jd[(jd.tamper == kind) & (jd.k == k)].set_index("monitor")
        a = audit_pooled(kind, k)
        cells = [name]
        for mon in ("rule", "judge", "strong"):
            cells.append(f"{g.loc[mon, 'auroc']:.2f}/{g.loc[mon, 'tpr_at_5pct_fpr']:.2f}" if mon in g.index else "--")
        cells.append("--" if np.isnan(a) else f"{a:.2f}/0")
        lines.append(" & ".join(cells) + r" \\")
    rows("tab_monitors_main", lines)

# ---------------------------------------------------------------- RQ3: recall by class, in-band vs off-log
if rq2 is not None:
    r2 = rq2[(rq2.selection == "random") & (rq2.k > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.4), sharey=True)
    for ax, cond, title in zip(axes, ("inband", "offlog"), ("in-band (sanctioned)", "off-log (unlogged)")):
        for kind in ("omission", "fabrication", "spoof", "mutual_forgery"):
            g = r2[(r2.condition == cond) & (r2.tamper == kind)].sort_values("k")
            if len(g):
                ax.errorbar(g.k, g.recall, yerr=[g.recall - g.recall_lo, g.recall_hi - g.recall], marker="o", capsize=2,
                            color=C[kind], label=NICE[kind])
        n = int(r2[r2.condition == cond].n_pos.max())
        ax.set_title(f"{title}, {n} positives")
        ax.set_xlabel("liars k (of 12)")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("recall (false-alarm rate 0.00)")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, "fig_recall_classes.png")

    lines = []
    for kind in ("omission", "fabrication", "spoof", "mutual_forgery"):
        cells = [NICE[kind]]
        for cond in ("inband", "offlog"):
            for k in (1, 2, 4, 8):
                g = r2[(r2.condition == cond) & (r2.tamper == kind) & (r2.k == k)]
                cells.append(fmt(g.recall.iloc[0]) if len(g) else "--")
        g = rq2[(rq2.tamper == kind) & (rq2.selection == "random")]
        cells.append(", ".join(f"{int(round(h.fpr.max() * h.n_neg.max()))}/{int(h.n_neg.max())}" for _, h in g.groupby("condition", sort=False)))
        lines.append(" & ".join(cells) + r" \\")
    rows("tab_recall", lines)
    lines = []
    for kind in ("omission", "fabrication", "spoof", "mutual_forgery"):
        for cond in ("inband", "offlog"):
            g = r2[(r2.condition == cond) & (r2.tamper == kind) & (r2.k == 4)]
            if len(g):
                r = g.iloc[0]
                lines.append(" & ".join([NICE[kind], cond, str(int(r.n_pos)), f"{r.recall:.2f}", f"{r.blame_precision:.2f}",
                                         f"{r.blame_recall:.2f}", f"{r.abstain:.2f}", f"{r.neg_tampered_alert:.2f}"]) + r" \\")
    rows("tab_blame", lines)

er = load("rq2_erasure_main.csv")
if er is not None:
    lines = []
    for topo in ("mesh", "star", "lanes", "star_targeted", "lanes_targeted"):
        cells = [topo.replace("_targeted", " (hub-targeted)")]
        for k in (2, 4, 6, 8):
            g = er[(er.topology == topo) & (er.k == k)]
            cells.append(f"{g.measured_erasable.iloc[0]:.2f} vs {g.theory.iloc[0]:.2f}" if len(g) else "--")
        lines.append(" & ".join(cells) + r" \\")
    rows("tab_erasure", lines)
    lines = []
    for topo, name in (("mesh", "mesh"), ("star", "star"), ("lanes", "lanes"), ("star_targeted", "star, hubs first"), ("lanes_targeted", "lanes, hubs first")):
        g = er[(er.topology == topo) & (er.k == 4)]
        if len(g):
            lines.append(" & ".join([name, f"{g.measured_erasable.iloc[0]:.2f}", f"{g.theory.iloc[0]:.2f}"]) + r" \\")
    rows("tab_erasure_main", lines)
for name in ("fig_erasure_main.png", "fig_recall_main.png"):
    if (RESULTS_DIR / name).exists():
        shutil.copy(RESULTS_DIR / name, FIG / name)
        print("->", FIG / name)

# ---------------------------------------------------------------- RQ1: census tables and zero counts
rq0 = load("rq0.csv")
if rq0 is not None:
    cond = {"honest": "honest", "gossip": "honest + gossip", "inband": "collusive, in-band", "offlog": "collusive, off-log",
            "stego": "collusive, stego", "paraphrase": "collusive, paraphrase", "stego_offlog": "collusive, off-log, encoded"}
    order = {"honest": 0, "gossip": 1, "inband": 2, "stego": 3, "paraphrase": 4, "offlog": 5, "stego_offlog": 6}
    tag_order = {"main": 0, "cond": 1, "stegooff": 2, "gemma": 3, "qwen": 4, "gemma26": 5}
    rq0 = rq0.assign(o=rq0.condition.map(order)).sort_values(["tag", "o"], key=lambda s: s.map(tag_order) if s.name == "tag" else s)
    lines, keep_lines = [], []
    keep = {("main", "honest"), ("cond", "gossip"), ("main", "inband"), ("main", "offlog"), ("gemma", "offlog"), ("qwen", "offlog")}
    census_model = {"cond": "deepseek-v4-flash", "stegooff": "deepseek-v4-flash"}
    for _, r in rq0.iterrows():
        alerts = int(round(r.alert_full * r.n))
        line = " & ".join([census_model.get(r.tag, MODEL.get(r.tag, r.tag)), cond.get(r.condition, r.condition), str(int(r.n)), f"{alerts}/{int(r.n)}",
                           f"{r.alert_msg_edges_only:.2f}", f"{r.alert_no_lastround_rule:.2f}", f"{r.nonowner_sightings_mean:.1f}",
                           f"{r.share_msg_path:.2f}" if r.nonowner_sightings_mean > 0 else "--",
                           f"{r.share_unexplained:.2f}" if r.nonowner_sightings_mean > 0 else "--"]) + r" \\"
        lines.append(line)
        if (r.tag, r.condition) in keep:
            keep_lines.append(line)
    rows("tab_census", lines)
    rows("tab_census_main", keep_lines)
    hon = rq0[rq0.condition.isin(["honest", "gossip"])]
    VALUES["num_honest"] = str(int(hon.n.sum()))
    print("honest episodes", int(hon.n.sum()), "alerts", float((hon.alert_full * hon.n).sum()))

    # RQ1 figure: where cross-agent sightings get their explanation, per condition
    sel = [("main", "honest", "honest"), ("cond", "gossip", "honest +\ngossip"), ("main", "inband", "in-band"),
           ("cond", "stego", "stego"), ("main", "offlog", "off-log"), ("stegooff", "stego_offlog", "off-log\nencoded"), ("gemma", "offlog", "off-log\n(gemma)")]
    fig, ax = plt.subplots(figsize=(8.4, 2.0))
    xs, msg, fil, une, lab = [], [], [], [], []
    for i, (tag, cnd, name) in enumerate(sel):
        r = rq0[(rq0.tag == tag) & (rq0.condition == cnd)]
        if not len(r):
            continue
        r = r.iloc[0]
        xs.append(len(xs))
        msg.append(r.share_msg_path if r.nonowner_sightings_mean > 0 else 0)
        fil.append(r.share_file_path if r.nonowner_sightings_mean > 0 else 0)
        une.append(r.share_unexplained if r.nonowner_sightings_mean > 0 else 0)
        lab.append((name, r.nonowner_sightings_mean, int(round(r.alert_full * r.n)), int(r.n)))
    msg, fil, une = np.array(msg), np.array(fil), np.array(une)
    ax.bar(xs, msg, 0.6, color="#2A6F97", label="explained by logged messages")
    ax.bar(xs, fil, 0.6, bottom=msg, color="#61A5C2", label="explained by a shared-file hop")
    ax.bar(xs, une, 0.6, bottom=msg + fil, color="#B5451B", label="no logged path (closure broken)")
    for x, (name, s, a, n) in zip(xs, lab):
        ax.text(x, 1.03, f"{s:.1f} sightings/ep\nalerts {a}/{n}", ha="center", fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([l[0] for l in lab], fontsize=7.5)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("share of sightings")
    ax.legend(loc="upper left", fontsize=6.5, ncol=3, frameon=False, bbox_to_anchor=(0, 1.32))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig_census.png")

# zero counts behind every FPR of 0.00: alerts on truthful negative logs per execution
lines = []
tot_neg = tot_alert = 0
for tag in ("main", "cond", "stegooff", "gemma", "qwen", "gemma26", "scale24", "scale48", "scale96", "long40", "adaptive_ds", "gen1", "gen2", "control", "adaptive"):
    d = load(f"pilot_report_{tag}.csv")
    if d is None:
        continue
    neg, pos = d[~d.positive], d[d.positive]
    tot_neg += len(neg)
    tot_alert += int(neg.alert.sum())
    ub = f"{3 / len(neg):.2f}" if len(neg) else "--"
    lines.append(" & ".join([MODEL.get(tag, tag), str(len(d)), str(len(neg)), str(int(neg.alert.sum())), ub, str(len(pos)),
                             f"{int(pos.alert.sum())}/{len(pos)}" if len(pos) else "--"]) + r" \\")
lines.append(r"\midrule")
lines.append(" & ".join([r"\textbf{all executions}", "", str(tot_neg), str(tot_alert), f"{3 / max(1, tot_neg):.3f}", "", ""]) + r" \\")
rows("tab_zeros", lines)
VALUES["num_negatives"] = str(tot_neg)

# ---------------------------------------------------------------- long horizon (40 rounds)
ql = load("rq2_long40.csv")
rl = load("rq3_long40.csv")
if ql is not None:
    ql = ql[ql.selection == "random"]
    lines = []
    for kind in ("omission", "fabrication", "spoof", "mutual_forgery"):
        cells = [NICE[kind]]
        for cond in ("inband", "offlog"):
            for k in (1, 2, 4):
                g = ql[(ql.condition == cond) & (ql.tamper == kind) & (ql.k == k)]
                cells.append(fmt(g.recall.iloc[0]) if len(g) else "--")
        g = ql[ql.tamper == kind]
        cells.append(", ".join(f"{int(round(h.fpr.max() * h.n_neg.max()))}/{int(h.n_neg.max())}" for _, h in g.groupby("condition", sort=False)))
        lines.append(" & ".join(cells) + r" \\")
    if rl is not None:
        r = rl[rl.level == "decode"].iloc[0]
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{8}{l}{tracer pair recall (decoder level): " + f"{r.recall:.2f} [{r.recall_lo:.2f}, {r.recall_hi:.2f}], lag {r.lag_rounds_mean:.1f} rounds, "
                     f"honest alerts {int(round(r.honest_episode_fpr * r.n_honest))}/{int(r.n_honest)}, false pairs {int(r.honest_false_pairs)}" + r"}\\")
    rows("tab_long40", lines)

# ---------------------------------------------------------------- RQ4: tracers
lines = []
for tag in ("main", "gemma", "qwen", "gemma26", "stegooff"):
    d = load(f"rq3_{tag}.csv")
    if d is None:
        continue
    for _, r in d.iterrows():
        n_h = int(r.n_honest) if not pd.isna(r.n_honest) else 0
        alerts = f"{int(round(r.honest_episode_fpr * n_h))}/{n_h}" if n_h and not pd.isna(r.honest_episode_fpr) else "--"
        lines.append(" & ".join([MODEL[tag], r.level, str(int(r.n_offlog_with_transfer)), f"{r.recall:.2f} [{r.recall_lo:.2f}, {r.recall_hi:.2f}]",
                                 fmt(r.lag_rounds_mean, 1), str(n_h), alerts, str(int(r.honest_false_pairs)) if n_h else "--"]) + r" \\")
if lines:
    rows("tab_tracers", lines)
    short = []
    for tag, name in (("main", "deepseek"), ("stegooff", "deepseek, encoded"), ("long40", "deepseek, 40 rounds"),
                      ("gemma", "gemma-31b"), ("gemma26", "gemma-26b"), ("qwen", "qwen-27b")):
        d = load(f"rq3_{tag}.csv")
        if d is None:
            continue
        r = d[d.level == "decode"].iloc[0]
        n_h = int(r.n_honest) if not pd.isna(r.n_honest) else 0
        alerts = f"{int(round(r.honest_episode_fpr * n_h))}/{n_h}" if n_h else "--"
        short.append(" & ".join([name, str(int(r.n_offlog_with_transfer)), f"{r.recall:.2f} [{r.recall_lo:.2f}, {r.recall_hi:.2f}]", fmt(r.lag_rounds_mean, 1), alerts]) + r" \\")
    rows("tab_tracers_main", short)
    # RQ4 figure: pair recall with intervals per run, lag annotated
    pts = []
    for tag, name in (("main", "deepseek"), ("stegooff", "deepseek\nencoded"), ("long40", "deepseek\n40 rounds"),
                      ("gemma", "gemma\n31b"), ("gemma26", "gemma\n26b"), ("qwen", "qwen\n27b")):
        d = load(f"rq3_{tag}.csv")
        if d is None:
            continue
        r = d[d.level == "decode"].iloc[0]
        pts.append((name, r.recall, r.recall_lo, r.recall_hi, r.lag_rounds_mean, int(r.n_offlog_with_transfer)))
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    x = np.arange(len(pts))
    ax.bar(x, [p[1] for p in pts], 0.6, yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], capsize=3,
           color=["#2A6F97", "#2A6F97", "#2A6F97", "#61A5C2", "#61A5C2", "#61A5C2"][: len(pts)])
    for xi, p in zip(x, pts):
        ax.text(xi, p[3] + 0.03, f"lag {p[4]:.1f}\nn={p[5]}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pts], fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("hidden pairs caught (recall)")
    ax.set_title("tracer detection of the unlogged cache", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig_tracers.png")
d = load("pilot_report_stegooff.csv")
if d is not None:
    VALUES["stegooff_credits"] = f"{d.credits.sum():.2f}"

lines = []
for suffix, lab in (("_exact", "exact match"), ("_decode", "with decoders")):
    d = load(f"rq2_cond{suffix}.csv")
    if d is None:
        continue
    for cond in ("stego", "paraphrase"):
        for kind in ("omission", "spoof", "mutual_forgery"):
            g = d[(d.condition == cond) & (d.tamper == kind) & (d.selection == "random")]
            if not len(g):
                continue
            cells = [cond, lab, NICE[kind], str(int(g.n_pos.max()))]
            for k in (1, 2, 4):
                h = g[g.k == k]
                cells.append(fmt(h.recall.iloc[0]) if len(h) else "--")
            cells.append(f"{int(round(g.fpr.max() * g.n_neg.max()))}/{int(g.n_neg.max())}")
            lines.append(" & ".join(cells) + r" \\")
if lines:
    rows("tab_hardcases", lines)

# ---------------------------------------------------------------- RQ5: views
v = load("rq4_main.csv")
if v is not None:
    order = ["peer_only", "gateway_only", "inference_only", "peer+gateway", "peer+inference", "all"]
    labels = ["peer\nonly", "tool gw\nonly", "inference\nonly", "peer +\ntool gw", "peer +\ninference", "all three"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.7))
    w = 0.26
    for ax, col, title in zip(axes, ("recall", "fpr"), ("recall (4 liars, omission)", "false-alarm rate on truthful logs")):
        for i, (content, lab) in enumerate((("full", "full content"), ("sample10", "10% content sample"), ("metadata", "metadata only"))):
            g = v[v.content == content].set_index("views").loc[order]
            x = np.arange(len(order)) + (i - 1) * w
            if col == "recall":
                ax.bar(x, g.recall, w, yerr=[g.recall - g.lo, g.hi - g.recall], capsize=1.5, label=lab, color=["#2A6F97", "#61A5C2", "#A9D6E5"][i])
            else:
                ax.bar(x, g.fpr, w, yerr=[0 * g.fpr, g.fpr_hi - g.fpr], capsize=1.5, label=lab, color=["#2A6F97", "#61A5C2", "#A9D6E5"][i])
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "fig_views.png")
    lines = []
    for name, lab in zip(order, ["peer only", "tool gateway only", "inference only", "peer + tool gateway", "peer + inference", "all three"]):
        for content in ("full", "sample10", "metadata"):
            r = v[(v.views == name) & (v.content == content)].iloc[0]
            lines.append(" & ".join([lab, content, f"{r.recall:.2f} [{r.lo:.2f}, {r.hi:.2f}]", f"{int(round(r.fpr * r.n_neg))}/{int(r.n_neg)}",
                                     f"{r.kb_per_episode:.0f}"]) + r" \\")
    rows("tab_views", lines)
    lines = []
    for name, content, lab in (("peer_only", "full", "peer logs only"), ("gateway_only", "full", "tool gateway only"),
                               ("peer+gateway", "metadata", "peer + tool gw, meta"), ("peer+inference", "metadata", "peer + inference, meta"),
                               ("all", "full", "all three, full")):
        r = v[(v.views == name) & (v.content == content)].iloc[0]
        lines.append(" & ".join([lab, f"{r.recall:.2f}", f"{int(round(r.fpr * r.n_neg))}/{int(r.n_neg)}", f"{r.kb_per_episode:.0f}"]) + r" \\")
    rows("tab_views_main", lines)
    # the same ablation under the other tamper classes (offline re-runs)
    lines = []
    for kind in ("omission", "spoof", "fabrication", "mutual_forgery"):
        d = v if kind == "omission" else load(f"rq4_main_{kind}.csv")
        if d is None:
            continue
        for name, lab in zip(order, ["peer only", "tool gateway only", "inference only", "peer + tool gw", "peer + inference", "all three"]):
            cells = [NICE[kind], lab]
            for content in ("full", "metadata"):
                r = d[(d.views == name) & (d.content == content)].iloc[0]
                cells.append(f"{r.recall:.2f}")
                cells.append(f"{int(round(r.fpr * r.n_neg))}/{int(r.n_neg)}")
            lines.append(" & ".join(cells) + r" \\")
    rows("tab_views_tamper", lines)

# ---------------------------------------------------------------- RQ2 extras: judge with gateway view, instructed labels
jg = load("rq1_main_judge_gateway.csv")
lines = []
if jg is not None:
    for _, r in jg[jg.monitor == "strong"].iterrows():
        lines.append(" & ".join([NICE.get(r.tamper, r.tamper), str(int(r.k)), str(int(r.n)), f"{r.auroc:.2f}", f"{r.tpr_at_5pct_fpr:.2f}"]) + r" \\")
    rows("tab_judge_gateway", lines)
lb = load("rq1_main_labels.csv")
if lb is not None:
    lines = []
    for _, r in lb.iterrows():
        lines.append(" & ".join([NICE.get(r.tamper, r.tamper), str(int(r.k)), r.monitor, r.label, str(int(r.n)), f"{r.auroc:.2f}", f"{r.tpr_at_5pct_fpr:.2f}"]) + r" \\")
    rows("tab_monitor_labels", lines)

# ---------------------------------------------------------------- scale
sc = load("scale.csv")
if sc is not None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.6))
    ax = axes[0]
    for kind in ("omission", "spoof"):
        for frac, ls in ((0.07, ":"), (0.33, "-")):
            g = sc[(sc.tamper == kind) & (sc.selection == "random") & (np.isclose(sc.liar_fraction, frac))].sort_values("n_agents")
            if len(g):
                ax.errorbar(g.n_agents, g.recall, yerr=[g.recall - g.recall_lo, g.recall_hi - g.recall], marker="o", ls=ls, capsize=2,
                            color=C[kind], label=f"{kind}, liars = {int(frac * 100)}% of swarm")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sc.n_agents.unique()))
    ax.set_xticklabels([str(int(n)) for n in sorted(sc.n_agents.unique())])
    ax.set_xlabel("swarm size n")
    ax.set_ylabel("recall (false-alarm rate 0.00)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("detection vs swarm size")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[1]
    g = sc[(sc.tamper == "omission") & (sc.selection == "random") & (np.isclose(sc.liar_fraction, 0.33))].sort_values("n_agents")
    for topo in ("mesh", "star", "lanes"):
        col = f"erasable_{topo}"
        if col in g:
            ax.plot(g.n_agents, g[col], marker="o", color=C[topo], label=f"measured, {topo}")
    ax.plot(g.n_agents, [random_share(int(n), int(k)) for n, k in zip(g.n_agents, g.k)], ls="--", color=C["theory"], label="Prop. 1 (random liars)")
    gt = sc[(sc.tamper == "omission") & (sc.selection == "targeted") & (np.isclose(sc.liar_fraction, 0.33))].sort_values("n_agents")
    for topo in ("star", "lanes"):
        col = f"erasable_{topo}"
        if col in gt:
            ax.plot(gt.n_agents, gt[col], marker="s", ls="--", color=C[topo], label=f"measured, {topo}, hub-targeted")
    ax.plot(gt.n_agents, [tree_worst_share(int(n), int(k)) for n, k in zip(gt.n_agents, gt.k)], ls=":", color=C["theory"], label="Prop. 2 (targeted, tree)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sc.n_agents.unique()))
    ax.set_xticklabels([str(int(n)) for n in sorted(sc.n_agents.unique())])
    ax.set_xlabel("swarm size n (liars = n/3)")
    ax.set_ylabel("share of messages erasable")
    ax.set_ylim(0, 1)
    ax.set_title("erasable share vs witness theory")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    fig.tight_layout()
    savefig(fig, "fig_scale.png")
    lines = []
    for _, r in sc.sort_values(["n_agents", "liar_fraction", "tamper", "selection"]).iterrows():
        lines.append(" & ".join([str(int(r.n_agents)), f"{int(r.liar_fraction * 100)}\\% ({int(r.k)})", r.tamper, r.selection, str(int(r.n_pos)),
                                 f"{r.recall:.2f} [{r.recall_lo:.2f}, {r.recall_hi:.2f}]", f"{int(round(r.fpr * r.n_neg))}/{int(r.n_neg)}",
                                 f"{r.credits_per_episode:.3f}"]) + r" \\")
    rows("tab_scale", lines)

# ---------------------------------------------------------------- RQ6: informed and learning adversary
g5 = load("rq5_vs_main.csv")
g5c = load("rq5_vs_control.csv")
if g5 is not None and "evasion_truthful_logs" in g5:
    arms = list(g5.arm)
    lab = {"baseline": "uninformed", "adaptive": "told rules\n(gen 0)", "gen1": "gen 1\n(outcomes)", "gen2": "gen 2\n(outcomes)"}
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.6))
    for ax, (col, lo, hi, title) in zip(axes, (("evasion_truthful_logs", "evasion_truthful_lo", "evasion_truthful_hi", "audit miss rate, truthful logs"),
                                              ("evasion_under_omission", "evasion_lo", "evasion_hi", "audit miss rate, colluders erase own records"),
                                              ("mission_success", "success_lo", "success_hi", "share of colluder tickets via fast path"))):
        x = np.arange(len(arms))
        ax.bar(x, g5[col], 0.6, yerr=[g5[col] - g5[lo], g5[hi] - g5[col]], capsize=3, color=["#9D9D9D", "#2A6F97", "#61A5C2", "#A9D6E5"][: len(arms)])
        ax.set_xticks(x)
        ax.set_xticklabels([lab.get(a, a) for a in arms], fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        for xi, (val, n) in enumerate(zip(g5[col], g5.n_collusive)):
            ax.text(xi, val + 0.04, f"{val:.2f}\n(n={int(n)})", ha="center", fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "fig_generations.png")
    lines = []
    for d, model in ((g5, "deepseek-v4-flash"), (g5c, "qwen3.5-397b")):
        if d is None or "evasion_truthful_logs" not in d:
            continue
        for _, r in d.iterrows():
            lines.append(" & ".join([model, r.arm, str(int(r.n_collusive)), str(int(r.inactive)),
                                     f"{r.evasion_truthful_logs:.2f} [{r.evasion_truthful_lo:.2f}, {r.evasion_truthful_hi:.2f}]",
                                     f"{r.evasion_under_omission:.2f} [{r.evasion_lo:.2f}, {r.evasion_hi:.2f}]",
                                     f"{r.mission_success:.2f}", f"{r.first_trip_round:.1f}"]) + r" \\")
    rows("tab_generations", lines)
    lines = []
    names = {"baseline": "uninformed", "adaptive": "rules shown", "gen1": "gen 1", "gen2": "gen 2"}
    for d, model in ((g5, "deepseek"), (g5c, "qwen3.5")):
        if d is None or "evasion_truthful_logs" not in d:
            continue
        for _, r in d.iterrows():
            arm = names.get(r.arm, r.arm) if model == "deepseek" else ("control" if r.arm == "baseline" else "rules shown")
            lines.append(" & ".join([f"{'ds' if model == 'deepseek' else 'qwen'}, {arm}", str(int(r.n_collusive)), f"{r.evasion_truthful_logs:.2f}", f"{r.evasion_under_omission:.2f}",
                                     f"{r.mission_success:.2f}", f"{r.first_trip_round:.1f}"]) + r" \\")
    rows("tab_generations_main", lines)

# ---------------------------------------------------------------- families (appendix)
fam = load("families.csv")
if fam is not None:
    lines = []
    for _, r in fam.iterrows():
        lines.append(" & ".join([MODEL.get(r.tag, r.tag), r.condition, str(int(r.n)), f"{r.accepted_24:.2f}", f"{r.positive_rate:.2f}",
                                 f"{r.alert_truthful_logs:.2f}", f"{r.trips_mean:.1f}", f"{r.hidden_mean:.1f}",
                                 fmt(r.rq2_omission_k4_recall), fmt(r.rq2_spoof_k4_recall), fmt(r.rq2_mutual_forgery_k4_recall),
                                 fmt(r.rq3_pair_recall), f"{r.credits_mean:.3f}"]) + r" \\")
    rows("tab_families", lines)

# ---------------------------------------------------------------- inline into main.tex
tex = MAIN.read_text(encoding="utf-8")
for name, body in ROWS.items():
    pat = re.compile(r"(% BEGIN ROWS " + re.escape(name) + r"\n).*?(% END ROWS " + re.escape(name) + r"\n)", re.S)
    if not pat.search(tex):
        print("no marker for", name)
        continue
    tex = pat.sub(lambda m: m.group(1) + body + m.group(2), tex)
for name, val in VALUES.items():
    pat = re.compile(r"^(\\def\\" + name.replace("_", "") + r"\{)[^}]*(\}% VALUE " + re.escape(name) + r")$", re.M)
    if not pat.search(tex):
        print("no value marker for", name)
        continue
    tex = pat.sub(lambda m: m.group(1) + val + m.group(2), tex)
with open(MAIN, "w", encoding="utf-8", newline="\n") as f:
    f.write(tex)
print("main.tex updated:", ", ".join(sorted(ROWS)), "|", ", ".join(f"{k}={v}" for k, v in VALUES.items()))
