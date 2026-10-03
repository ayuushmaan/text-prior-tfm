"""Unified text-heavy tabular benchmark: STRABLE + OpenML (+ TabArena members with text).

Protocol (matches the marimo notebook):
- stratified sample <= --rows, binary targets (majority-vs-rest if multiclass)
- ExtraTrees numeric-only vs numeric + TF-IDF(300) text, 3-fold CV, clean + masked numerics

STRABLE part reuses scripts/benchmark_strable.py. OpenML part needs the
registry below (task id, text columns) - filled from the feature-type survey
(see README_TFM3.md "text-heavy survey").

Usage:
  python scripts/benchmark_text_tables.py --source strable [--rows 800] [--mask 0.5]
  python scripts/benchmark_text_tables.py --source openml --tables <name..> [--rows 800]
  python scripts/benchmark_text_tables.py --list
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_strable import TABLES as STRABLE_TABLES  # noqa: E402
from benchmark_strable import evaluate as et_evaluate  # noqa: E402
from benchmark_strable import load as load_strable  # noqa: E402

# Survey result (123 tables: 51 TabArena + 72 CC18): only 6 TabArena tables have
# ANY string column, all short codes/dates. Single text-ish member: Diabetes.
OPENML_TABLES = {
    "coupon": {"task": 363681, "text_cols": ["coupon"],
               "note": "TabArena; 5 coupon labels; ET +0.070/+0.050"},
    "coil": {"task": 363624, "text_cols": ["avgAge", "contributionPrivateThirdPartyInsurance"],
             "note": "TabArena; age-bucket labels; ET -0.004/+0.000 (no signal)"},
    "diabetes": {"task": 363630,
                 "text_cols": ["medical_specialty", "diag_1", "diag_2", "diag_3"],
                 "note": "TabArena; 70 specialties + ICD codes; ET +0.055/-0.009"},
}

TABARENA_TEXT_TASKS: list = [
    # The only TabArena members with >=1 string column (6/51, all short codes/dates):
    363618,  # bank-marketing (poutcome)
    363624,  # coil2000 (age buckets)
    363679,  # hr-job-change (experience numbers)
    363681,  # in-vehicle-coupon (labels)
    363684,  # marketing-campaign (dates)
    363700,  # seismic-bumps (letters)
]


def load_openml(name, n_sample=800, seed=0):
    import openml
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    spec = OPENML_TABLES[name]
    task = openml.tasks.get_task(spec["task"], download_splits=False)
    ds = task.get_dataset(download_data=True)
    target = spec.get("target", task.target_name)
    X, y, _, _ = ds.get_data(target=target, dataset_format="dataframe")
    tcols = [c for c in spec["text_cols"] if c in X.columns]
    # Python-level join: pandas>=3 Arrow strings keep pd.NA through astype(str)
    T = [" | ".join("" if pd.isna(v) else str(v) for v in row)
         for row in X[tcols].itertuples(index=False, name=None)]
    Xn = X.drop(columns=tcols)
    Xn = Xn.apply(pd.to_numeric, errors="coerce")
    Xn = Xn.fillna(Xn.median(numeric_only=True)).to_numpy(float)
    y = np.asarray(y)
    if y.dtype == object or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y.astype(str))
    y = np.asarray(y)
    if len(np.unique(y)) > 2:  # majority-vs-rest binarization, documented per table
        top = np.bincount(y).argmax()
        y = (y == top).astype(int)
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    k = min(n_sample // 2, len(i0), len(i1))
    ii = np.concatenate([rng.choice(i0, k, replace=False), rng.choice(i1, k, replace=False)])
    rng.shuffle(ii)
    return Xn[ii].astype(np.float32), y[ii].astype(np.int64), [T[i] for i in ii]


def run_one(name, X, y, T, mask):
    b, w = et_evaluate(X, y, T, 0.0, 0)
    bm, wm = et_evaluate(X, y, T, mask, 0)
    print(f"[{name}] n={len(y)} clean: num={b:.3f} +txt={w:.3f} (d={w-b:+.3f}) | "
          f"mask={mask}: num={bm:.3f} +txt={wm:.3f} (d={wm-bm:+.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["strable", "openml", "carte"], default="strable")
    ap.add_argument("--tables", nargs="*", default=None)
    ap.add_argument("--rows", type=int, default=800)
    ap.add_argument("--mask", type=float, default=0.5)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("strable:", sorted(STRABLE_TABLES))
        print("openml:", sorted(OPENML_TABLES))
        print("tabarena-text-tasks:", TABARENA_TEXT_TASKS)
        return
    if args.source == "carte":
        from benchmark_carte import main as _mc

        sys.argv = ["benchmark_carte.py", "--rows", str(args.rows), "--mask", str(args.mask)]
        _mc()
        return
    if args.source == "strable":
        from benchmark_strable import main as _m  # reuse wholesale

        sys.argv = ["benchmark_strable.py", "--rows", str(args.rows), "--mask", str(args.mask)]
        _m()
        return
    names = args.tables or sorted(OPENML_TABLES)
    for name in names:
        X, y, T = load_openml(name, args.rows)
        run_one(name, X, y, T, args.mask)


if __name__ == "__main__":
    main()
