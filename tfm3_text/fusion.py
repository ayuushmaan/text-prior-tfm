"""Fusion: frozen text embeddings -> extra columns for a tabular backbone.

Backbone-agnostic by design: both NanoTabPFN (TFM-Playground) and
NanoTabICLv2 (soda-inria/nanotabicl) consume plain numeric matrices, so
projected text columns can be concatenated to X without touching the
backbone. The projection is fixed (seeded) for ablations; replace
``project_text_to_cols`` with a learned ``nn.Linear`` for pretraining.
"""

import numpy as np


def project_text_to_cols(E, n_cols=2, seed=0, scale=0.1):
    """Fixed random projection of text embeddings to n_cols extra features."""
    rng = np.random.RandomState(seed)
    P = rng.randn(E.shape[1], n_cols).astype(np.float64)
    return (np.asarray(E, dtype=np.float64) @ P * scale).astype(np.float32)


def fuse_numeric_text(X_num, X_cat, E_text, n_text_cols=2, seed=0):
    """Concatenate numeric + categorical + projected-text columns."""
    parts = [np.asarray(X_num, dtype=float)]
    if X_cat is not None and np.asarray(X_cat).size:
        parts.append(np.asarray(X_cat, dtype=float))
    parts.append(project_text_to_cols(np.asarray(E_text, dtype=float), n_text_cols, seed))
    return np.hstack(parts).astype(np.float32)
