"""Hub-tier text-heavy benchmark: public HF datasets beyond inria-soda.

Currently: vietnam-real-estates (3.5M rows, name/description/location + price).
Scan notes (see README): Hub tabular search surfaced few parquet-native mixed
tables; tabula-8b-eval-suite is numeric eval tables (skip), Amazon meta has no
target (skip), Lichess PGNs are niche. vietnam is the Tier-3 anchor.

Needs: pandas, scikit-learn, scipy, huggingface_hub, pyarrow.
Usage: python scripts/benchmark_hub.py [--rows 800] [--mask 0.5]
"""

import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

VN_NUM = ["area", "floor_count", "frontage_width", "house_depth", "road_width",
          "bedroom_count", "bathroom_count"]
VN_TXT = ["name", "description", "property_type_name", "province_name", "district_name"]


def num(s):
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float).copy()
    v[np.isnan(v)] = np.nanmedian(v)
    return v


def load_vietnam(n_sample=800, seed=0):
    df = pd.read_parquet("hf://datasets/tinixai/vietnam-real-estates/shard_0000.parquet")
    X = np.stack([num(df[c]) for c in VN_NUM], 1)
    T = [" ".join("" if pd.isna(v) else str(v) for v in row)
         for row in df[VN_TXT].itertuples(index=False, name=None)]
    p = pd.to_numeric(df["price"], errors="coerce").to_numpy(float)
    m = ~np.isnan(p)
    y = (p[m] >= float(np.median(p[m]))).astype(int)
    X, T = X[m], [T[i] for i in np.where(m)[0]]
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    k = min(n_sample // 2, len(i0), len(i1))
    ii = np.concatenate([rng.choice(i0, k, replace=False), rng.choice(i1, k, replace=False)])
    rng.shuffle(ii)
    return X[ii].astype(np.float32), y[ii].astype(np.int64), [T[i] for i in ii]


def evaluate(X, y, T, mask_frac, seed):
    rng = np.random.RandomState(seed)
    Xm = X.copy().astype(float)
    if mask_frac:
        Xm[rng.rand(*Xm.shape) < mask_frac] = 0.0
    base = cross_val_score(ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
                           Xm, y, cv=3).mean()
    E = TfidfVectorizer(max_features=300).fit_transform(T)
    with_text = cross_val_score(ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
                                hstack([csr_matrix(Xm), E]).tocsr(), y, cv=3).mean()
    return base, with_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=800)
    ap.add_argument("--mask", type=float, default=0.5)
    args = ap.parse_args()
    X, y, T = load_vietnam(args.rows)
    b, w = evaluate(X, y, T, 0.0, 0)
    bm, wm = evaluate(X, y, T, args.mask, 0)
    print(f"[vietnam] n={len(y)} clean: num={b:.3f} +txt={w:.3f} (d={w-b:+.3f}) | "
          f"mask={args.mask}: num={bm:.3f} +txt={wm:.3f} (d={wm-bm:+.3f})", flush=True)


if __name__ == "__main__":
    main()
