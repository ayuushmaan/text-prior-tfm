# Application kit — TFM research roles

## Template recommendation: Awesome-CV (LaTeX)

For a company working on tabular foundation models, you are hired on
**research taste + evidence**. Use **[Awesome-CV](https://github.com/posquit0/Awesome-CV)**
(`posquit0/Awesome-CV` — also in the Overleaf gallery, so no local LaTeX needed).

Why this one:
- The default for ML research CVs; reviewers skim it in seconds.
- Handles what matters for you: preprints, project bullets with numbers,
  open-source links. No photo, no gimmicks, ATS-readable.
- One page if you can (you can: lead with TFM3.0), two max.

Runner-up: RenderCV (YAML-in, modern PDF-out) if you hate LaTeX. Avoid:
Canva/Docs visual templates (signal junior; break ATS parsing).

## Structure (in this order)

1. **Header** — name, email, GitHub, Scholar/arxiv link, location.
2. **Research interests** — one line, e.g. "tabular foundation models,
   synthetic priors, multimodal tables".
3. **Publications & preprints** — TFM3.0 paper (status: in preparation,
   link repo/preprint). Even "in preparation" counts if the repo is public.
4. **Research experience / projects** — TFM3.0 first, with quantified bullets
   (see `cv.md`). Older work after, shorter.
5. **Open source** — repo links with stars/checkpoints if any.
6. **Skills** — PyTorch, HuggingFace, OpenML, experiment discipline
   (ablations, controls, seeds, artifacts).
7. **Education** — degrees, thesis if relevant.

## The bar for a TFM company

They will open your repo before finishing your cover letter. Make sure:
- `TFM3.0/README_TFM3.md` reads as a project landing page (it nearly does).
- `results/` has the JSONs (done), `paper/draft.md` exists (done).
- Pin the repo on GitHub with a one-paragraph description + the fig2 PNG.

Files here: `cv.md` (content skeleton — paste into Awesome-CV),
`cover_letter.md` (tailored skeleton).
