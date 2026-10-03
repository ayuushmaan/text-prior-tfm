# Coherent Synthetic Text as a Fallback Channel for Tabular Foundation Models

**TFM3.0 — research draft v0.1** (results frozen 2026-10-03; artifacts in `../results/`)

## Abstract

Tabular foundation models (TFMs) such as TabPFN and TabICL are pretrained
exclusively on synthetic *numeric* priors, yet real tables are full of text.
We ask whether a TFM can be given a text channel **without any real text**:
we augment the TabICL-v2 synthetic prior with *coherent text fields* —
deterministic, row-grounded verbalizations of each synthetic row — and train a
nano-scale TFM with a frozen sentence encoder plus a small learned projector.
The text is redundant by construction (a pure function of the inputs), so any
gain must come from how the model uses it, not from new information.

Findings across two nano backbones (NanoTabICLv2, NanoTabPFN), two projector
widths (2, 8), template and LLM-paraphrased text, and 11 real text-heavy
tables: (1) the generator is faithful by construction (categorical keyword
coverage 1.00, numeric recoverability AUC 0.96); (2) at small scale (32–48
tables) the text channel is a **shortcut**: it fits training loss best and
generalizes worst; (3) at 500 tables with masked pretraining the gap **flips**:
held-out +0.007 clean / +0.087 under 50% masking, reproduced in an independent
rerun with checkpoints; (4) a shuffled-text control with identical capacity
contributes exactly nothing (−0.007), proving the gain is grounding, not
parameters; (5) in-context accuracy scales monotonically with context length
(1k→4k) on NanoTabPFN; (6) on real tables the synthetically-trained text
pathway transfers as a **fallback circuit**: it helps under masking on 4 of 7
transfer tables (up to +0.10) while TF-IDF references confirm the text signal
is real (up to +0.32). A survey of 123 OpenML/TabArena tables finds them
effectively text-free, motivating our 11-table text-heavy benchmark
(STRABLE + OpenML + CARTE + Hub), released as scripts.

## 1. Introduction

Tabular foundation models pretrained on synthetic priors (TabPFN,
Hollmann et al.; TabICL/TabICLv2, Qu et al.) recently dethroned
gradient-boosted trees on tabular benchmarks. Their pretraining data,
however, is purely numeric (categoricals appear only as integer codes).
Real tables — clinical notes, product descriptions, accident narratives,
reviews — carry free text that is often the most predictive signal.

Existing text-aware tabular work fuses text from *real* tables
(STRABLE/T4 pretraining, per-table adapters) or post-hoc adaptation. No
prior work injects coherent text into the *synthetic prior itself* and
pretrains the in-context learner on it. We do exactly this, with a fully
open, lightweight recipe: template verbalizer → frozen MiniLM → learned
linear projector → extra columns for any numeric-matrix TFM backbone.

Contributions:
1. **A leakage-controlled coherent-text generator** for SCM priors, with a
   rank-percentile mapping fix (naive scaling saturates) and a slot verifier.
2. **Evidence for a shortcut-to-fallback transition**: negative at small
   scale, positive at 500 tables, with a shuffled-text control closing the
   capacity objection.
3. **Transfer evidence**: the synthetic text pathway helps on real text-heavy
   tables under missingness, plus an 11-table benchmark and a negative
   survey result (TabArena/CC18 are text-free).
4. **Open artifacts**: extension library (`tfm3_text/`), benchmark scripts,
   pools, checkpoints, and result JSONs.

## 2. Related work

**Table-native ICL + semantics.** ConTextTab (SAP, NeurIPS 2025) adds MiniLM
modality embeddings to a PFN-style learner trained on real T4 tables — the
closest hybrid; we use synthetic text instead. The TabPFN Text Adapter
(Fischer et al. 2026) freezes MiniLM + TabPFN-2.5 and trains only an MLP
adapter — our direct blueprint; we port the idea to TabICL-v2/nano and
pretrain on synthetic rather than real text. TabSTAR (NeurIPS 2025) shows
*unfrozen* e5 wins at higher cost — our frozen-vs-unfrozen comparison point.
TIME (Luo et al. 2025) gives the frozen-TFM + missingness evaluation
protocol we adopt.

**Names help.** TabLLM (AISTATS 2023) serializes rows as "The col is value"
and proves semantics help few-shot prediction — we steal its template
philosophy. TransTab (NeurIPS 2022) contributes the name+value tokenizer
with a separate numeric path. TP-BERTa (ICLR 2024) shows naive
number-as-string fails, motivating our rank-percentile mapping. CARTE
(ICML 2024, same lab as TabICL) proves frozen word vectors + tabular
transformers work, with reusable checkpoints and 51 datasets.

**Synthetic tables with semantics.** GReaT (ICLR 2023, `be_great`) fine-tunes
small LMs on serialized rows — our paraphrase path. TapTap (EMNLP 2023) is
the only work proving synthetic-from-table-LM rows boost downstream tabular
models. TabSyn/TabDDPM/CTSyn exclude free text entirely. No work augments an
SCM *prior* with text for TFM pretraining.

**Benchmarks.** STRABLE (Blayer et al. 2026, 108 raw-string tables, same SODA
lab) finds TF-IDF+TabICL-v2 Pareto-optimal — our primary eval target and
reference baseline. CARTE-51 and the AutoML Multimodal benchmark cover
entity-string and long-text regimes.

## 3. Method

### 3.1 Semantic grounding generator

A prior table has anonymous columns `x_0..x_{d-1}` and integer codes.
Per table we sample a domain (clinic, loan); the schema maps columns to
(name, unit, plausible range) and codes to words. Numeric mentions use
**per-column rank percentiles** mapped into range: raw SCM values are
heavy-tailed and naive scaling collapses all rows to one phrase (observed:
every row became "70.6-year-old"). Templates render one sentence per row.
The text is a function of X only — never the label. Missing slots render
`?` instead of raising (all-numeric tables). An LLM stage (Qwen2.5-0.5B)
optionally paraphrases CSV rows; a slot verifier keeps only faithful rows
(1,888/2,048 = 92.2% accepted; template↔Qwen cosine 0.63).

### 3.2 Frozen encoder + fusion

Text → frozen `all-MiniLM-L6-v2` (384-d; TF-IDF variant for ablations) →
learned `Linear(384, k)`, `k ∈ {2, 8}` → concatenated as extra columns.
Backbone-agnostic: identical trick for NanoTabICLv2 and NanoTabPFN, both of
which consume numeric matrices. The encoder never trains; at 500 tables the
projector is the only text-side capacity.

### 3.3 Training protocol

Paired runs: control (numeric-only) vs fusion with **identical init**
(the backbones share grouped feature embeddings, so widths differ safely),
identical batches, AdamW 3e-4, mask-train fraction 0.3 on numerics (text
intact) unless stated. Context 64/36 or 48/16 train/test rows. Held-out
tables are unseen SCM draws; real-table transfer uses 400-shot in-context
evaluation, clean and 50%-masked.

## 4. Experimental setup

**Priors.** Main pools: simplified TabICL-v2 prior (`nanotabicl`, 3 num +
2 cat cols, binary, 64 rows); verification batches from the full
`tabicl.prior.PriorDataset` on CUDA. **Backbones.** NanoTabICLv2
(embed 32, 1/1/2 blocks) and NanoTabPFN from TFM-Playground
(embed 64, 2 layers) for the context study. **Benchmarks.** 11 text-heavy
tables: STRABLE chocolate/osha/beer, OpenML coupon/coil/diabetes, CARTE
ebert/whisky/coffee/yelp, Hub vietnam-real-estates — each with an
ExtraTrees numeric vs +TF-IDF reference (clean + masked) and, for 7,
TFM transfer deltas.

## 5. Results

### 5.1 The generator is faithful

Categorical keyword coverage 1.00; median-split numeric recoverability from
text AUC 0.96. Qwen paraphrase fidelity 0.96 on 5 rows (verifier caught a
male→female flip a naive substring check missed) and 0.92 at 2,048-row
scale. Greedy 0.5B output is stylistically rigid ("Age is X, Gender is
Y…") — faithful but not diverse; diversity needs sampling temperature or a
larger model.

### 5.2 Shortcuts at small scale, flip at 500 tables

With 32–48 tables, fusion fits train loss best (0.12–0.48 vs 0.44–0.58)
yet generalizes worst (e.g. clean 0.664 vs 0.797). With 192 tables +
text-dropout it reaches parity (+0.002/+0.005). With **500 tables**
(400 train / 100 held-out, 1,500 steps): train 0.47/0.39; held-out clean
**0.796→0.812 (+0.016)**, masked **0.698→0.786 (+0.089)**. An independent
rerun with saved checkpoints confirms: +0.007 / +0.087
(`results/results_500_box3.json`).

### 5.3 The gain is grounding, not capacity

A shuffled-text arm (identical 8 columns/params, row permutation breaking
grounding): clean 0.798 (−0.006 vs control), masked 0.704 (−0.007), while
grounded text gives +0.007/+0.087 (`results/results_500_3way.json`). The
capacity objection is closed.

### 5.4 ICL scales with context on NanoTabPFN

Trained on 100-row tables, tested at 1k/2k/3k/4k context (10–40× length
extrapolation): control accuracy 0.690→0.694→0.696→0.700, monotone and
saturating; memory 0.46→2.2 GB. The fusion arm trails (−0.09), consistent
with §5.2 at small effective diversity.

### 5.5 Real-table transfer: a fallback circuit

TF-IDF references confirm large real text signals (up to +0.32 on coffee).
The synthetic-trained pathway (never saw real text) helps **under masking**
on 4/7 tables: choc +0.020, osha +0.053, whisky +0.100, yelp +0.073; clean
transfer is neutral-to-negative, and fails where numerics dominate (beer)
or capacity is insufficient (coffee: TF-IDF 0.85 vs TFM ~0.50 with 1 numeric
col). Full table in README; per-table scripts in `scripts/`.

### 5.6 The benchmark and the negative survey

Scanning 123 OpenML/TabArena tables (51 TabArena + 72 CC18) plus string-length
audits: only 6/51 TabArena tables contain any string column — all short
codes, buckets, numbers-as-strings, or dates. CC18 adds nothing (DNA
letters, binary fingerprints, short codes). The sole text-ish member is
Diabetes130US/medical_specialty. A Hub-wide scan of top tabular datasets
likewise surfaced mostly numeric/code tables (tabula-8b-eval-suite is
numeric; Amazon meta lacks targets); the anchor find is vietnam-real-estates
(3.5M rows, ET +0.141/+0.200). Conclusion: mainstream tabular benchmarks are
effectively text-free, which is both a gap statement and a justification for
our 11-table suite (`scripts/benchmark_{strable,text_tables,carte,hub}.py`).

## 6. Discussion and limitations

Everything is small-scale by design (tiny backbones, ≤1,500 steps, ≤500
tables, 0.5B paraphraser):directional, not SOTA-chasing. The redundant-text
setup isolates mechanism at the cost of realism — real text is
non-redundant, which is exactly where transfer works. Failure modes are
reported, not hidden: beer (strong numerics), coffee (projector capacity),
Qwen rigidity, the pandas-3 Arrow string gotchas in the scripts. Checkpoints
and pools are versioned per run; sandbox-ephemeral runs are marked as such.

## 7. Conclusion

Coherent synthetic text does not lift clean-data accuracy by adding
information — it can't, being a function of X. What it does, at sufficient
table diversity, is install a **fallback circuit** for degraded numerics:
+0.087 masked at 500 tables, all of it grounding, transferring to real
text-heavy tables. That, plus the generator, the benchmark, and the
text-free audit of mainstream suites, is the contribution.

## Reproducibility

`tfm3_text/` (generator, encoder, fusion, eval) runs on CPU with torch +
sklearn; `scripts/smoke_text.py` and `scripts/pretrain_text.py` verify the
pipeline without heavy deps. GPU runs mirror `scripts/gpu_playground_text.py`;
headline numbers are frozen in `results/results_500_box3.json` and
`results/results_500_3way.json`. Full TabICL prior needs `pip install
tabicl`; the playground harness (`upstream/TFM-Playground`) needs Python 3.12.

## Key references

TabICL (Qu et al., ICML 2025) / TabICLv2 (arXiv:2602.11139, soda-inria/tabicl);
nanoTabPFN (Pfefferle et al., arXiv:2511.03634) / TFM-Playground
(automl/TFM-Playground); TabPFN Text Adapter (Fischer et al., arXiv:2606.04876);
ConTextTab (Spinaci et al., NeurIPS 2025); TabSTAR (Arazi et al., NeurIPS 2025);
CARTE (Kim et al., ICML 2024); STRABLE (Blayer et al., arXiv:2605.12292);
TabLLM (Hegselmann et al., AISTATS 2023); GReaT (Borisov et al., ICLR 2023);
TransTab (Wang & Sun, NeurIPS 2022); TP-BERTa (Yan et al., ICLR 2024);
TIME (Luo et al., arXiv:2506.00813); TabuLa-8B (Gardner et al., NeurIPS 2024).
