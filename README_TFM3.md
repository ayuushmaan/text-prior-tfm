# TFM3.0 - coherent text fields for TabICL-v2 synthetic tables

Research project: extend a TabICL-v2 generator base with **coherent text fields** and train/evaluate a **nano TFM** with a frozen text encoder + fusion.

## Layout

- `upstream/TFM-Playground/` - untouched clone of [automl/TFM-Playground](https://github.com/automl/TFM-Playground) (reference harness: `TabICLPriorDataLoader`, `PriorDumpDataLoader`, `train()`, `NanoTabPFNModel`).
- `tfm3_text/` - our extension (light deps only: torch, sklearn, scipy, numpy):
  - `semantic_schema.py` - domain schemas (clinic, loan) grounding anonymous columns to names/units/words.
  - `text_generator.py` - rank-mapped verbalizer + `augment_playground_batch()` (adds `"texts"` to any playground batch dict).
  - `text_encoder.py` - frozen TF-IDF encoder (swap for MiniLM/E5 later, same interface).
  - `fusion.py` - text embeddings -> extra columns (backbone-agnostic: NanoTabPFN *or* NanoTabICLv2).
  - `evaluate_text.py` - coherence metrics + ExtraTrees ablations (full-data and missing-numeric).
- `scripts/smoke_text.py` - local CPU smoke test (no tabicl install needed).
- `scripts/pretrain_text.py` - paired control-vs-fusion pretraining (tabicl or sklearn prior, tfidf/minilm).
- `scripts/gpu_playground_text.py` - GPU reference flow (mirrors the marimo notebook).
- `configs/base.yaml` - shared hyper-parameters.
- `paper/draft.md` - full paper draft (v0.1) with all results; `results/summary.json` - headline numbers.

## Key design decisions

1. **Text = function of X only, never y** (no label leakage). Numeric mentions use per-column rank percentiles mapped into plausible ranges (raw SCM values are heavy-tailed; naive scaling saturates).
2. **Frozen encoder + fusion** (MiniLM/E5 or TF-IDF projected as extra columns). Keeps the tabular prior intact; text can't distort numeric learning.
3. **Backbone-agnostic**: TFM-Playground trains `NanoTabPFNModel`; the marimo notebook also tests `NanoTabICLv2` (soda-inria/nanotabicl). Fusion works with either since both consume numeric matrices.

## Findings (marimo notebook, RTX PRO 6000)

- Coherence: categorical keyword coverage **1.00**, numeric recoverability AUC **0.96**.
- Full data (ExtraTrees judge): numeric 0.965 vs +text 0.935 - text is redundant when X is clean (expected).
- **Missing numerics: text wins** - 50% masked +7.2pts (0.828 -> 0.900), 90% masked +13.7pts (0.733 -> 0.870).
- Real `tabicl.prior.PriorDataset` batch on `cuda:0` verbalized end-to-end (34x3 table).
- NanoTabICLv2 forward on GPU ~6ms (warm).
- Paired pretraining (box 1, mlp_scm, 48 tables, MiniLM-frozen + learned Linear(384,2), 300 steps): train loss fuse **0.48** vs ctrl **0.58**, but held-out ICL acc ctrl **0.64** vs fuse **0.58** - fusion overfits table-specific text cues with only 48 tables (shortcut learning). Gap closes under 90% masking (-0.002); shuffled-text ablation shows grounding worth only +0.01. Train-time masking (0.3) does not flip the sign.
- Scale baseline (box 2, mix_scm control): held-out 0.688 clean / 0.674 masked.
- Takeaway: redundant text (pure function of X) adds capacity but no signal at test time; next runs need (a) 4-8x more tables, (b) text-dropout/two-stage projector training, (c) eval on STRABLE free-text-led where text carries non-redundant signal.
- With 192 tables + text-dropout (box 2): fusion reaches parity (clean +0.002, masked +0.005) - diversity direction confirmed.
- **Text-heavy transfer (box 1, 3 STRABLE tables, 400-shot ICL, models trained ONLY on synthetic text)** - ExtraTrees ref: text helps everywhere (+0.03/+0.06). TFM pair: choc clean -0.025 / masked **+0.020**; osha clean -0.005 / masked **+0.053**; beer clean -0.065 / masked -0.042 (strong-numeric regime, projector mismatch). I.e. synthetic-text pretraining transfers a fallback circuit that helps under masking on 2/3 real text-heavy tables. See `scripts/benchmark_strable.py`.
- **OpenML/TabArena text survey (123 tables: 51 TabArena + 72 CC18, metadata scan + string-length audit)**: only 6/51 TabArena tables have ANY string column and all are short codes/dates (bank poutcome, coil buckets, HR experience numbers, coupon labels, Marketing dates, seismic letters); CC18's nominal-heavy tables are DNA letters, binary fingerprints, or short codes. Sole text-ish member: Diabetes130US/medical_specialty (70 values, avg len 16). TabArena/CC18 are effectively text-free; true text-heavy evaluation lives on STRABLE/CARTE/AMLB.
- **Unified 6-table benchmark** (`scripts/benchmark_text_tables.py`): STRABLE choc/osha/beer + OpenML coupon/coil/diabetes. ET numeric vs +TF-IDF (clean/masked): coupon +0.070/+0.050, coil -0.004/+0.000, diabetes +0.055/-0.009.
- **CARTE tier (4 tables, `scripts/benchmark_carte.py`)**: ebert ET -0.010/+0.029, TFM -0.145/-0.035; whisky ET -0.013/+0.071, TFM -0.152/**+0.100**; coffee ET **+0.315**/+0.315, TFM +0.010/-0.045; yelp ET +0.074/+0.100, TFM -0.003/**+0.073**. Pattern: clean-transfer hurts on rich-text tables (2-dim synthetic projector + distribution shift), fallback helps under masking (whisky, yelp). Coffee (1 numeric col) shows the capacity gap: TF-IDF 0.85 vs TFM ~0.50.
- **Hub scan**: top-120 Hub tabular datasets are mostly numeric/codes; tabula-8b-eval-suite is numeric eval tables (skip), Amazon meta has no target (skip). Anchor found: **vietnam-real-estates** (3.5M rows, descriptions + locations + price, `scripts/benchmark_hub.py`): ET +0.141/+0.200, TFM -0.083/-0.042.
- **Qwen paraphrase path** (`scripts/paraphrase_qwen.py`, Qwen2.5-0.5B-Instruct): CSV+schema -> grounded sentences at 24/25 slot fidelity (0.96; verifier catches flips like male->female), template-vs-Qwen cosine 0.62. Full 2048-row pool distilling on box 2.
- **500-table scale run (box 1, 400 train / 100 held-out, 1500 steps, mask-train 0.3)**: train ctrl 0.47 / tmpl8 0.39; held-out clean ctrl 0.796 -> tmpl8 **0.812 (+0.016)**, masked 0.698 -> **0.786 (+0.089)**. The gap FLIPS POSITIVE at scale - first clean win for fusion. Shortcuts fade with table diversity as predicted.
- **Rerun with artifacts (box 3, same recipe, checkpoints every 500 + `results/results_500_box3.json`)**: clean ctrl 0.804 -> tmpl8 0.811 (+0.007), masked 0.711 -> 0.798 (+0.087). Flip confirmed; artifacts: `pool500.npz`, `ckpt_500_step{500,1000,1500}.pt` (see notebook cell outputs for paths).
- **Shuffled-text control (same 8 extra cols, row grounding broken, `results/results_500_3way.json`)**: clean ctrl 0.804 / tmpl8 0.811 / shuf8 0.798; masked 0.711 / 0.798 / 0.704. The full masked gain (+0.087) is grounding; ungrounded capacity contributes nothing (-0.007). Capacity objection closed.
- Prior small-scale context (box 2, 32 tables, 8-dim, 3 arms): train ctrl 0.44 / tmpl8 0.19 / qwen8 0.12, but held-out clean ctrl 0.797 > tmpl8 0.688 > qwen8 0.664; masked 0.703 > 0.648 = 0.648. At small scale redundant synthetic text creates shortcuts (qwen fits best, generalizes worst) - the contrast with the 500-table flip is the core scaling finding.

## Gotchas

- `PriorDataset` default `n_jobs=-1` crashes on Python 3.13 (unpicklable `HpSampler...sub_sampler` closure). Use `n_jobs=1` (or `loader.pd.n_jobs = 1` with the playground wrapper).
- Full `TFM-Playground` install needs **Python ==3.12** + heavy extras (`tabicl[pretrain]`, `ticl`, `pfns`, `schedulefree`...). The `tfm3_text` extension intentionally avoids all of that.
- Keep long prior-generation jobs in background threads on hosted notebook services (foreground cells get interrupted).

## Quickstart

```bash
# lightweight extension only (local, CPU)
py -3.11 -m pip install -r requirements_text.txt
py -3.11 scripts/smoke_text.py

# full playground harness (Linux + CUDA + Python 3.12)
pip install -e upstream/TFM-Playground
python -m tfmplayground.external_priors --lib tabicl --prior_type mix_scm \
    --num_batches 1000 --batch_size 4 --min_features 3 --max_features 3 \
    --max_seq_len 50 --max_classes 3 --save_path tabicl_4k_50x3.h5
```
