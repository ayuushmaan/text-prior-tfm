# Cover letter skeleton — TFM company

Dear [TODO: hiring manager / team],

I work on tabular foundation models — specifically on giving TFMs a text
channel without relying on real text. In my current project (TFM3.0,
[repo link]), I augment the TabICL-v2 synthetic prior with coherent,
row-grounded text fields and pretrain nano-scale models with a frozen
sentence encoder plus a learned projector.

Two results may interest you directly: first, the text channel only starts
helping at 500 tables (+0.007 clean / +0.089 under masking) after hurting at
small scale — a shortcut-to-fallback transition with a shuffled-text control
ruling out capacity confounds. Second, the synthetically-trained text pathway
transfers to real text-heavy tables under missingness (up to +0.10 on
STRABLE/CARTE tables), which speaks to [TODO: company's] work on [TODO:
their text/multimodal direction — cite a paper or product line].

I also surveyed 123 OpenML/TabArena tables and found them effectively
text-free, and built the 11-table benchmark I wish had existed. Everything is
open and reproduced with saved checkpoints, including a Qwen-paraphrase
distillation path with verifier-measured fidelity.

I would welcome the chance to discuss [TODO: specific role/team challenge].
Thank you for your time.

Best regards,
[TODO: Name]
