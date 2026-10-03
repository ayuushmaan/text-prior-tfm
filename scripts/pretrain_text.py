"""Paired pretraining: numeric-only control vs frozen-text fusion (mirrors box-1 run).

Needs: torch, scikit-learn, scipy, numpy. Prior: tabicl (pip install tabicl)
or --prior sklearn (stand-in, CPU smoke). Encoder: --encoder tfidf (default,
no downloads) or minilm (needs sentence-transformers or transformers).

Example (CPU smoke):  py -3.11 scripts/pretrain_text.py --prior sklearn --steps 30
Example (GPU, real):  python scripts/pretrain_text.py --prior tabicl --prior-type mlp_scm --steps 300 --mask-train 0.3
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tfm3_text.evaluate_text import coherence_report  # noqa: E402
from tfm3_text.fusion import project_text_to_cols  # noqa: E402
from tfm3_text.text_encoder import FrozenTfidfEncoder  # noqa: E402
from tfm3_text.text_generator import make_text_dataset  # noqa: E402


def build_pool(args):
    if args.prior == "sklearn":
        from sklearn.datasets import make_classification

        Xs, ys, Es_texts = [], [], []
        for t in range(args.tables):
            Xn, y = make_classification(n_samples=args.rows, n_features=args.features,
                                        n_informative=args.features, n_redundant=0,
                                        n_classes=args.classes, random_state=1000 + t)
            data = make_text_dataset(Xn, None, domain_name=args.domain, seed=t)
            Xs.append(Xn.astype(np.float32))
            ys.append(y.astype(np.int64))
            Es_texts.append(data["texts"])
        enc = FrozenTfidfEncoder(max_features=args.encoder_dim)
        Eflat = enc.fit_transform([t for texts in Es_texts for t in texts])
        E = Eflat.reshape(args.tables, args.rows, -1)
        return np.stack(Xs), np.stack(ys), E
    # tabicl prior (serial-safe: n_jobs=1)
    from tabicl.prior import PriorDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pd = PriorDataset(regression=False, batch_size=args.tables, batch_size_per_gp=4,
                      min_features=args.features, max_features=args.features,
                      max_classes=args.classes, min_seq_len=args.rows,
                      max_seq_len=args.rows + 8, prior_type=args.prior_type,
                      n_jobs=1, device="cpu")
    x, y, _, _, _ = next(pd)
    S = min(args.rows, x.shape[1])
    X = x[:, :S, : args.features].float().numpy().astype(np.float32)
    Y = y[:, :S].long().numpy().astype(np.int64)
    all_texts, per = [], []
    for t in range(args.tables):
        data = make_text_dataset(X[t], None, domain_name=args.domain, seed=t)
        per.append(data["texts"])
        all_texts.extend(data["texts"])
    if args.encoder == "minilm":
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)
        Eflat = st.encode(all_texts, batch_size=256, show_progress_bar=False,
                          convert_to_numpy=True).astype(np.float32)
    else:
        Eflat = FrozenTfidfEncoder(max_features=args.encoder_dim).fit_transform(all_texts)
    return X, Y, Eflat.reshape(args.tables, S, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["sklearn", "tabicl"], default="sklearn")
    ap.add_argument("--prior-type", default="mlp_scm")
    ap.add_argument("--encoder", choices=["tfidf", "minilm"], default="tfidf")
    ap.add_argument("--encoder-dim", type=int, default=128)
    ap.add_argument("--tables", type=int, default=64)
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--features", type=int, default=4)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--domain", default="clinic")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--mask-train", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from tfm3_text.nanobackend import make_nano  # optional vendored backbone
    except Exception:
        make_nano = None
    if make_nano is None:
        print("nanobackend not vendored; see notebook (nanomodel.py) for the TFM backbone.")
        print("Running pipeline check (generation + coherence) only.")
    X, Y, E = build_pool(args)
    data0 = make_text_dataset(X[0], None, domain_name=args.domain, seed=0)
    rep = coherence_report(data0["texts"], X[0], None, data0["domain"], data0["cat_names"])
    print(f"pool: X={X.shape} E={E.shape} enc={args.encoder} coherence={rep}")
    print("PIPELINE OK (full paired training runs in the GPU notebook; port backbone to run here)")


if __name__ == "__main__":
    main()
