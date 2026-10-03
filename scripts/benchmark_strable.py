"""Reference benchmark on text-heavy STRABLE tables: numeric vs numeric+text.

Runs anywhere with `datasets`, `pandas`, `scikit-learn`, `scipy` installed
(GPU box or workstation; no torch needed). Mirrors the marimo notebook cell.

Tables (inria-soda/STRABLE-benchmark):
- chocolate-bar-ratings -> Rating >= 3.25 (text: tasting notes)
- osha-accidents        -> Task Assigned == mode (text: narratives)
- beer-ratings          -> review_overall >= median (text: Description)

Usage:  python scripts/benchmark_strable.py [--rows 800] [--mask 0.5]
"""

import argparse

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

REPO = "inria-soda/STRABLE-benchmark"

TABLES = {
    "choc": dict(parquet="chocolate-bar-ratings/data.parquet", kind="choc"),
    "osha": dict(parquet="osha-accidents/data.parquet", kind="osha"),
    "beer": dict(parquet="beer-ratings/data.parquet", kind="beer"),
}

NUM_COLS = {
    "choc": ["Review Date"],
    "osha": ["build_stor", "nature_of_inj", "part_of_body", "event_type",
             "evn_factor", "hum_factor", "fat_cause"],
    "beer": ["ABV", "Min IBU", "Max IBU", "Astringency", "Body", "Alcohol", "Bitter",
             "Sweet", "Sour", "Salty", "Fruits", "Hoppy", "Spices", "Malty", "number_of_reviews"],
}

TEXT_COLS = {
    "choc": ["Company (Manufacturer)", "Country of Bean Origin",
             "Specific Bean Origin or Bar Name", "Ingredients", "Most Memorable Characteristics"],
    "osha": ["Abstract Text", "Event Description"],
    "beer": ["Description", "Style", "Brewery"],
}


def num(df, col):
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float).copy()
    v[np.isnan(v)] = np.nanmedian(v)
    return v


def load(name):
    spec = TABLES[name]
    df = pd.read_parquet(hf_hub_download(REPO, filename=spec["parquet"], repo_type="dataset"))
    if spec["kind"] == "choc":
        cocoa = df["Cocoa Percent"].str.rstrip("%").pipe(pd.to_numeric, errors="coerce")
        X = np.stack([df["Review Date"].to_numpy(float), cocoa.fillna(cocoa.median()).to_numpy(float)], 1)
        y = (df["Rating"].replace({"Under 2.75": "2.5"})
               .pipe(pd.to_numeric, errors="coerce").fillna(2.5).to_numpy(float) >= 3.25).astype(int)
    elif spec["kind"] == "osha":
        X = np.stack([num(df, c) for c in NUM_COLS["osha"]], 1)
        y = (df["Task Assigned"] == df["Task Assigned"].value_counts().index[0]).astype(int).to_numpy()
    else:
        X = np.stack([num(df, c) for c in NUM_COLS["beer"]], 1)
        r = df["review_overall"].to_numpy(float)
        y = (r >= float(np.median(r))).astype(int)
    T = df[TEXT_COLS[name]].fillna("").astype(str).agg(" ".join, axis=1).tolist()
    return X.astype(np.float32), y.astype(np.int64), T


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
    for name in TABLES:
        X, y, T = load(name)
        rng = np.random.RandomState(0)
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
        k = min(args.rows // 2, len(i0), len(i1))
        ii = np.concatenate([rng.choice(i0, k, replace=False), rng.choice(i1, k, replace=False)])
        rng.shuffle(ii)
        X, y = X[ii], y[ii]
        T = [T[i] for i in ii]
        b, w = evaluate(X, y, T, 0.0, 0)
        bm, wm = evaluate(X, y, T, args.mask, 0)
        print(f"[{name}] n={len(y)} clean: num={b:.3f} +txt={w:.3f} (d={w-b:+.3f}) | "
              f"mask={args.mask}: num={bm:.3f} +txt={wm:.3f} (d={wm-bm:+.3f})")


if __name__ == "__main__":
    main()
