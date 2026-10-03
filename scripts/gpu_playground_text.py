"""GPU reference flow (run on the marimo/CUDA box, mirrors the notebook cells).

1. pip install tabicl   (base package is enough for tabicl.prior.PriorDataset)
2. git clone --depth 1 https://github.com/automl/TFM-Playground.git /tmp/TFM-Playground
3. python scripts/gpu_playground_text.py  (needs torch+cuda, sklearn, scipy)

NOTE: on Python 3.13 set PriorDataset(n_jobs=1) - the default multiprocessing
pool hits an unpicklable-sampler bug (HpSampler...sub_sampler). The
playground wrapper TabICLPriorDataLoader does not expose n_jobs, so either
instantiate PriorDataset directly (below) or set ``loader.pd.n_jobs = 1``.
"""

import importlib.util
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/tmp/TFM-Playground")
sys.path.insert(0, "tfm3_text/..")  # local extension when copied next to the box

from tfm3_text.evaluate_text import coherence_report
from tfm3_text.text_generator import augment_playground_batch, make_text_dataset


def load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "pg_tabicl", "/tmp/TFM-Playground/tfmplayground/external_priors/tabicl.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    from tabicl.prior import PriorDataset

    pd = PriorDataset(
        regression=False, batch_size=1, batch_size_per_gp=1,
        min_features=3, max_features=3, max_classes=2,
        min_seq_len=20, max_seq_len=40, prior_type="mlp_scm", n_jobs=1, device="cpu",
    )
    t0 = time.time()
    x, y, _, _, train_size = next(pd)
    dt = time.time() - t0
    print(f"TabICL-v2 batch: x={tuple(x.shape)} y={tuple(y.shape)} split={train_size} took={dt:.1f}s")

    xc = x.cuda()
    torch.cuda.synchronize()
    print("on", xc.device, "|", torch.cuda.get_device_name(0))

    data = make_text_dataset(x[0].float().numpy(), None, domain_name="clinic", seed=0)
    for t in data["texts"][:3]:
        print("-", t)
    rep = coherence_report(
        data["texts"], x[0].float().numpy(), None, data["domain"], data["cat_names"]
    )
    print("coherence:", {k: round(v, 3) if isinstance(v, float) else v for k, v in rep.items()})


if __name__ == "__main__":
    main()
