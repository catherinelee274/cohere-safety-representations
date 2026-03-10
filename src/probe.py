"""
Probe training utilities.

Two probe types:
  1. DiffMean direction   — mean(harmful) - mean(harmless), L2-normalised
     Used as a fast, interpretable linear direction (no training data needed).

  2. Logistic Regression  — sklearn LR trained on (activations, labels)
     Reports accuracy, AUC-ROC, and the weight vector.

Both return a ProbeResult dataclass for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    model_key:   str
    lang:        str
    layer:       int
    probe_type:  str          # "diffmean" | "logreg"
    direction:   np.ndarray   # shape (hidden,) — the probe direction
    accuracy:    float
    auc:         float
    # Per-fold results for cross-validated probes
    fold_accs:   list[float] = field(default_factory=list)
    fold_aucs:   list[float] = field(default_factory=list)
    # Raw scores for every sample (for silhouette / alignment)
    scores:      Optional[np.ndarray] = None


# ── DiffMean ──────────────────────────────────────────────────────────────────

def diffmean_probe(
    acts:       np.ndarray,   # (N, hidden)
    labels:     np.ndarray,   # (N,) — 0 = harmless, 1 = harmful
    model_key:  str = "",
    lang:       str = "",
    layer:      int = -1,
) -> ProbeResult:
    """
    Compute the mean-difference direction and evaluate as a linear classifier.
    Direction = mean(harmful) - mean(harmless), L2-normalised.
    Classification threshold = 0 (sign of projection).
    """
    harmful_mean  = acts[labels == 1].mean(0)
    harmless_mean = acts[labels == 0].mean(0)
    direction     = harmful_mean - harmless_mean
    norm          = np.linalg.norm(direction)
    if norm < 1e-8:
        direction = direction
    else:
        direction = direction / norm

    scores  = acts @ direction
    preds   = (scores > scores.mean()).astype(int)
    acc     = accuracy_score(labels, preds)
    auc     = roc_auc_score(labels, scores)

    return ProbeResult(
        model_key=model_key, lang=lang, layer=layer,
        probe_type="diffmean",
        direction=direction,
        accuracy=acc, auc=auc,
        scores=scores,
    )


# ── Logistic Regression ───────────────────────────────────────────────────────

def logreg_probe(
    acts:       np.ndarray,
    labels:     np.ndarray,
    model_key:  str = "",
    lang:       str = "",
    layer:      int = -1,
    n_folds:    int = 5,
    C:          float = 1.0,
    max_iter:   int = 1000,
) -> ProbeResult:
    """
    Train a logistic regression probe with stratified k-fold CV.
    Fits a final probe on all data and returns its weight vector.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(acts)

    fold_accs, fold_aucs = [], []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X, labels):
        clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        clf.fit(X[train_idx], labels[train_idx])
        preds  = clf.predict(X[val_idx])
        probs  = clf.predict_proba(X[val_idx])[:, 1]
        fold_accs.append(accuracy_score(labels[val_idx], preds))
        fold_aucs.append(roc_auc_score(labels[val_idx], probs))

    # Final probe on all data
    clf_full = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf_full.fit(X, labels)
    direction = clf_full.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    scores = X @ direction
    all_probs = clf_full.predict_proba(X)[:, 1]

    return ProbeResult(
        model_key=model_key, lang=lang, layer=layer,
        probe_type="logreg",
        direction=direction,
        accuracy=float(np.mean(fold_accs)),
        auc=float(np.mean(fold_aucs)),
        fold_accs=fold_accs,
        fold_aucs=fold_aucs,
        scores=all_probs,
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def silhouette_score_probe(
    acts:   np.ndarray,
    labels: np.ndarray,
) -> float:
    """Silhouette score on the raw activation space (no PCA)."""
    from sklearn.metrics import silhouette_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    # Sub-sample for speed if large
    if len(acts) > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(acts), 2000, replace=False)
        acts, labels = acts[idx], labels[idx]
    return float(silhouette_score(acts, labels, metric="cosine"))


def cosine_alignment(dir1: np.ndarray, dir2: np.ndarray) -> float:
    """Cosine similarity between two probe directions."""
    d1 = dir1 / (np.linalg.norm(dir1) + 1e-8)
    d2 = dir2 / (np.linalg.norm(dir2) + 1e-8)
    return float(np.dot(d1, d2))
