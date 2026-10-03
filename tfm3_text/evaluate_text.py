"""Evaluation: generator coherence (goal 3) + prediction lift (goal 1).

Coherence metrics (reference values from the marimo notebook, clinic domain):
- categorical keyword coverage ~1.00 (every text mentions its row's words)
- numeric recoverability AUC ~0.96 (median-split of col0 predictable from text)

Prediction ablations use ExtraTrees (fast, no GPU) as a neutral judge:
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from tfm3_text.fusion import fuse_numeric_text
from tfm3_text.text_encoder import FrozenTfidfEncoder


def coherence_report(texts, X_num, X_cat, domain, cat_names, max_features=500):
    texts = list(texts)
    Xn = np.asarray(X_num, dtype=float)
    Xc = np.zeros((len(texts), 0), dtype=int) if X_cat is None else np.asarray(X_cat, dtype=int)
    hits = []
    for i, t in enumerate(texts):
        tl = t.lower()
        h = 0
        for j, name in enumerate(cat_names):
            opts = domain["cat_features"].get(name, [])
            w = opts[int(Xc[i, j]) % len(opts)].lower() if len(opts) else ""
            h += bool(w) and w in tl
        hits.append(h / max(1, len(cat_names)))
    vec = FrozenTfidfEncoder(max_features=max_features)
    E = vec.fit_transform(texts)
    target = (Xn[:, 0] > np.median(Xn[:, 0])).astype(int)
    auc = cross_val_score(LogisticRegression(max_iter=500), E, target, cv=3, scoring="roc_auc").mean()
    return {"cat_coverage": float(np.mean(hits)), "num_recover_auc": float(auc), "vocab": vec.dim}


def _base_matrix(X_num, X_cat):
    parts = [np.asarray(X_num, dtype=float)]
    if X_cat is not None and np.asarray(X_cat).size:
        parts.append(np.asarray(X_cat, dtype=float))
    return np.hstack(parts).astype(float)


def ablation_numeric_vs_text(X_num, X_cat, y, texts, seed=0, max_features=300):
    Xb = _base_matrix(X_num, X_cat)
    y = np.asarray(y)
    base = cross_val_score(
        ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1), Xb, y, cv=3
    ).mean()
    E = FrozenTfidfEncoder(max_features=max_features).fit_transform(texts)
    fused = hstack([csr_matrix(Xb), csr_matrix(E)]).tocsr()
    with_text = cross_val_score(
        ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1), fused, y, cv=3
    ).mean()
    return {"numeric_only": float(base), "with_text": float(with_text), "delta": float(with_text - base)}


def ablation_missing_numeric(X_num, X_cat, y, texts, drop_frac=0.5, seed=0, max_features=300):
    """Zero-mask numeric entries (text still carries them) -> text should help."""
    rng = np.random.RandomState(seed)
    Xm = np.asarray(X_num, dtype=float).copy()
    Xm[rng.rand(*Xm.shape) < drop_frac] = 0.0
    return {
        "drop_frac": drop_frac,
        **ablation_numeric_vs_text(Xm, X_cat, y, texts, seed=seed, max_features=max_features),
    }
