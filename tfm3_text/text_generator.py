"""Coherent text generation from synthetic tabular rows.

Design rules (leakage-controlled, see README_TFM3.md):
1. Text is a function of X only, never of y.
2. Numeric mentions use per-column *rank percentiles* mapped into the
   schema's plausible range, so text spans the full semantic range instead
   of saturating (raw SCM values are heavy-tailed).
3. Missing slots (e.g. all-numeric tables vs. templates with categorical
   slots) render as "?" instead of raising.
"""

import random

import numpy as np

from tfm3_text.semantic_schema import ground_columns


class _Default(dict):
    def __missing__(self, key):
        return "?"


def rank_fracs(X):
    """Per-column rank percentile in [0, 1]. X: (n_rows, n_cols) array."""
    X = np.asarray(X, dtype=float)
    order = np.argsort(np.argsort(X, axis=0), axis=0)
    return order / max(1, X.shape[0] - 1)


def verbalize_row(col_fracs, cat_row, domain, num_names, cat_names, rng):
    """Render one row. col_fracs: rank percentiles; cat_row: int codes."""
    mapping = {}
    for name, frac in zip(num_names, col_fracs):
        spec = next((f for f in domain["num_features"] if f[0] == name), None)
        if spec is None:
            mapping[name] = f"{float(frac):.2f}"
        else:
            _, _, lo, hi = spec
            mapping[name] = f"{lo + float(frac) * (hi - lo):.1f}"
    for j, name in enumerate(cat_names):
        opts = domain["cat_features"].get(name, ["a", "b"])
        mapping[name] = opts[int(cat_row[j]) % len(opts)] if len(cat_row) else opts[0]
    mapping["entity"] = domain["entity"]
    return rng.choice(domain["templates"]).format_map(_Default(mapping))


def make_text_dataset(X_num, X_cat=None, domain_name="clinic", seed=0):
    """Numpy arrays -> list of coherent text fields.

    Returns dict with texts + schema (mirrors the marimo notebook demo).
    """
    X_num = np.asarray(X_num, dtype=float)
    n_rows, n_num = X_num.shape
    X_cat = np.zeros((n_rows, 0), dtype=int) if X_cat is None else np.asarray(X_cat, dtype=int)
    fracs = rank_fracs(X_num)
    domain, num_names, cat_names = ground_columns(n_num, X_cat.shape[1], domain_name)
    rng = random.Random(seed)
    texts = [
        verbalize_row(fracs[i].tolist(), X_cat[i].tolist(), domain, num_names, cat_names, rng)
        for i in range(n_rows)
    ]
    return {"texts": texts, "num_names": num_names, "cat_names": cat_names, "domain": domain}


def augment_playground_batch(batch, domain_name="clinic", seed=0, table_index=0):
    """Add coherent ``"texts"`` to a TFM-Playground batch dict.

    Works with batches from ``TabICLPriorDataLoader`` (full TabICL-v2 prior),
    ``PriorDumpDataLoader`` (HDF5 dumps) or any dict with an ``"x"`` tensor of
    shape (batch, seq_len, n_features). All columns are treated as numeric
    (TabICL prior tables carry no categorical metadata); use ``make_text_dataset``
    directly if integer categorical codes are available.
    Returns a *new* dict; input batch is not mutated.
    """
    x = batch["x"][table_index].detach().cpu().float().numpy()
    out = dict(batch)
    out["texts"] = make_text_dataset(x, None, domain_name, seed)["texts"]
    out["text_table_index"] = table_index
    return out
