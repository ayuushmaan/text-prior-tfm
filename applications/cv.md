# [TODO: Full Name]
[TODO: email] · [TODO: phone] · [GitHub](https://github.com/[TODO]) · [Google Scholar](https://scholar.google.com/[TODO]) · [TODO: City]

## Research interests
Tabular foundation models, synthetic data priors, multimodal (text + table) learning.

## Publications & preprints
- **[TODO: Surname] et al. "Coherent Synthetic Text as a Fallback Channel for Tabular
  Foundation Models."** *In preparation*, 2026. — Synthetic-text augmentation of the
  TabICL-v2 prior; +0.087 held-out accuracy under 50% masking with a shuffled-text
  control; 11-table text-heavy benchmark. [repo](https://github.com/[TODO]/TFM3.0) · [draft](paper/draft.md)

## Research experience

### Coherent Text for Tabular Foundation Models (TFM3.0) — Independent research, 2026
- Extended the TabICL-v2 synthetic prior with a leakage-controlled text generator
  (rank-percentile verbalizer; categorical coverage 1.00, numeric recoverability AUC 0.96).
- Pretrained nano TFMs (NanoTabICLv2 / NanoTabPFN) with a frozen MiniLM + learned
  projector, paired controls, identical init, seeded and checkpointed runs.
- Showed a scale flip: text hurts at 32 tables (shortcut learning) but wins at 500
  tables (+0.007 clean / +0.089 masked); a shuffled-text control attributes 100% of
  the gain to grounding, ruling out a capacity confound.
- Demonstrated transfer: the synthetic-trained text pathway helps on 4/7 real
  text-heavy tables under masking (up to +0.10); surveyed 123 OpenML/TabArena tables
  and found them effectively text-free, motivating an 11-table benchmark (STRABLE +
  OpenML + CARTE + Hub) with scripted ExtraTrees references.
- Stack: PyTorch (CUDA), HuggingFace Transformers/Datasets, OpenML, scikit-learn;
  2-GPU experiment protocol across hosted notebooks with background-job orchestration.

### [TODO: previous project / internship] — [TODO: org], [TODO: dates]
- [TODO: one quantified bullet]
- [TODO: one quantified bullet]

## Open source
- **TFM3.0** — text-augmented tabular priors + benchmark suite (PyTorch).
  `tfm3_text/` library, 6 benchmark scripts, saved pools/checkpoints/JSON results.

## Technical skills
- **ML**: PyTorch, HuggingFace (Transformers/Datasets/Hub), scikit-learn, OpenML API.
- **Practice**: ablation design, paired controls, seed discipline, checkpointed/artifacts-first
  experimentation, scaling studies (100→500 tables, 1k→4k ICL context).
- **Languages**: Python, [TODO: others].

## Education
- **[TODO: Degree], [TODO: University]** — [TODO: years]. [TODO: thesis/honors, one line.]
