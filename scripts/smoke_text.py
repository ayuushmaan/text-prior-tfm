"""Local CPU smoke test: no heavy deps (no tabicl install, no GPU needed).

Uses sklearn's make_classification as a stand-in SCM prior to exercise the
full text pipeline: grounding -> generation -> coherence -> ablations.
The same functions run unchanged on real TabICL-v2 batches (see
scripts/gpu_playground_text.py and the marimo notebook).

Run:  py -3.11 scripts/smoke_text.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tfm3_text.evaluate_text import ablation_missing_numeric, ablation_numeric_vs_text, coherence_report
from tfm3_text.text_generator import make_text_dataset


def main():
    rng = np.random.RandomState(0)
    Xn, y = make_classification(
        n_samples=400, n_features=3, n_informative=3, n_redundant=0,
        n_classes=2, random_state=0,
    )
    Xc = (rank_like(Xn[:, :2]) * 2).astype(int)  # 2 pseudo-categorical codes
    data = make_text_dataset(Xn, Xc, domain_name="clinic", seed=1)
    print("sample:", data["texts"][0])

    rep = coherence_report(data["texts"], Xn, Xc, data["domain"], data["cat_names"])
    print(f"coherence: cat_coverage={rep['cat_coverage']:.3f} "
          f"num_recover_auc={rep['num_recover_auc']:.3f} vocab={rep['vocab']}")

    abl = ablation_numeric_vs_text(Xn, Xc, y, data["texts"])
    print(f"full-data: numeric={abl['numeric_only']:.3f} +text={abl['with_text']:.3f} "
          f"delta={abl['delta']:+.3f}")

    for frac in (0.5, 0.9):
        m = ablation_missing_numeric(Xn, Xc, y, data["texts"], drop_frac=frac)
        print(f"missing {int(frac * 100)}%: numeric={m['numeric_only']:.3f} "
              f"+text={m['with_text']:.3f} delta={m['delta']:+.3f}")

    # sanity gates (thresholds calibrated on sklearn stand-in, not TabICL batches)
    assert rep["cat_coverage"] > 0.9, rep
    assert rep["num_recover_auc"] > 0.7, rep
    print("SMOKE OK")


def rank_like(col):
    order = np.argsort(np.argsort(col))
    return order / max(1, len(col) - 1)


if __name__ == "__main__":
    main()
