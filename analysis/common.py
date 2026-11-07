"""Shared analysis helpers: episode loading, labels, bootstrap intervals, AUROC and TPR at fixed FPR."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.io import list_episodes, load_episode, truth_label  # noqa: E402
from config import RESULTS_DIR, RUNS_DIR  # noqa: E402


def load_all(tag=None, runs_dir=RUNS_DIR):
    eps = []
    for d in list_episodes(runs_dir):
        # exact tag match: episode dirs are <tag>_<topology>_<condition>_s<seed>, so a prefix test alone
        # would let "adaptive" also pick up "adaptive_ds_*"
        if tag and not any(d.name.startswith(f"{tag}_{t}_") for t in ("mesh", "star", "lanes")):
            continue
        ep = load_episode(d)
        ep["label"] = truth_label(ep)
        eps.append(ep)
    return eps


def bootstrap_ci(values, stat=np.mean, n=2000, seed=0, alpha=0.05):
    """Episode-level bootstrap: values is one number per episode."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(stat(values)), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels, dtype=int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(scores, labels, fpr=0.01):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    neg, pos = scores[labels == 0], scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thresh = np.quantile(neg, 1 - fpr, method="higher")
    return float(np.mean(pos > thresh))


def save_table(rows, name):
    import pandas as pd
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / name
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("->", out)
    return df
