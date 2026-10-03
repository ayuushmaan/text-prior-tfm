"""CARTE-tier text-heavy benchmark (inria-soda/carte-benchmark, raw tables).

Tables (all targets pre-binarized 0/1 upstream except whisky Meta_Critic):
- roger_ebert (movies: names/directors/actors/genre)
- whisky (names + tasting-note clusters)
- coffee_ratings (full review texts desc_1/2/3)
- yelp (names + categories + city; 64k rows, sampled)

Needs: datasets, pandas, scikit-learn, scipy, huggingface_hub.
Usage: python scripts/benchmark_carte.py [--rows 800] [--mask 0.5]
"""

import argparse

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

REPO = "inria-soda/carte-benchmark"
TABLES = ["roger_ebert", "whisky", "coffee_ratings", "yelp"]


def num(s):
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float).copy()
    v[np.isnan(v)] = np.nanmedian(v)
    return v


def texts(df, cols):
    return [" ".join("" if pd.isna(v) else str(v) for v in row)
            for row in df[cols].itertuples(index=False, name=None)]


def load(name):
    df = pd.read_parquet(hf_hub_download(REPO, filename=f"data_carte/{name}/raw.parquet",
                                         repo_type="dataset"))
    if name == "roger_ebert":
        X = np.stack([num(df["year"]), num(df["duration"])], 1)
        T = texts(df, ["movie_name", "directors", "actors", "genre"])
        y = df["critic_rating"].to_numpy(float)
    elif name == "whisky":
        cost = df["Cost"].str.extract(r"(\d+)")[0].pipe(pd.to_numeric, errors="coerce")
        X = cost.fillna(cost.median()).to_numpy(float).reshape(-1, 1)
        T = texts(df, ["Whisky", "Cluster", "Country", "Type", "Class"])
        y = df["Meta_Critic"].to_numpy(float)
        y = (y >= float(np.nanmedian(y))).astype(int)
    elif name == "coffee_ratings":
        X = pd.factorize(df["roast"])[0].reshape(-1, 1).astype(float)
        T = texts(df, ["desc_1", "desc_2", "desc_3", "roaster", "name", "origin"])
        y = df["rating"].to_numpy(float)
    else:  # yelp
        X = np.stack([num(df["latitude"]), num(df["longitude"]), num(df["review_count"]),
                      num(df["number_of_days_open"]),
                      df["price_range"].fillna("").astype(str).str.len().to_numpy(float)], 1)
        T = texts(df, ["name", "categories", "city"])
        y = df["stars"].to_numpy(float)
        y = (y >= float(np.nanmedian(y))).astype(int)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(y)
    uq = np.unique(y[m])
    yb = y[m].astype(int) if set(uq.tolist()) <= {0.0, 1.0} else (y[m] >= float(np.median(y[m]))).astype(int)
    return X[m].astype(np.float32), yb.astype(np.int64), [T[i] for i in np.where(m)[0]]


def sample(X, y, T, n=800, seed=0):
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    k = min(n // 2, len(i0), len(i1))
    ii = np.concatenate([rng.choice(i0, k, replace=False), rng.choice(i1, k, replace=False)])
    rng.shuffle(ii)
    return X[ii], y[ii], [T[i] for i in ii]


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
        X, y, T = sample(*load(name), n=args.rows)
        b, w = evaluate(X, y, T, 0.0, 0)
        bm, wm = evaluate(X, y, T, args.mask, 0)
        print(f"[{name}] n={len(y)} clean: num={b:.3f} +txt={w:.3f} (d={w-b:+.3f}) | "
              f"mask={args.mask}: num={bm:.3f} +txt={wm:.3f} (d={wm-bm:+.3f})", flush=True)


if __name__ == "__main__":
    main()
