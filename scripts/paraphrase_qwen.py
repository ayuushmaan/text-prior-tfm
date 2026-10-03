"""Qwen paraphrases of synthetic tabular rows (mirrors the box-1 notebook demo).

Flow: TabICL-v2 (or sklearn stand-in) rows -> CSV + schema prompt -> Qwen
(Qwen2.5-0.5B-Instruct, GPU) rewrites each row as one grounded sentence ->
slot verifier checks every value survived (catches flips like male->female).

Needs: transformers, torch+cuda. Model downloads ~1GB on first run.
Usage: python scripts/paraphrase_qwen.py [--rows 5]
"""

import argparse
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tfm3_text.text_generator import rank_fracs
except ImportError:  # allow running from scripts/ dir directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tfm3_text.text_generator import rank_fracs

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DOMAIN = {"entity": "patient",
          "num": [("age", 18, 90), ("blood_pressure", 90, 180), ("cholesterol", 120, 300)],
          "cat": {"gender": ["female", "male"], "smoker": ["non-smoker", "smoker"]}}


def display_rows(Xn, Xc):
    fracs = rank_fracs(np.asarray(Xn, dtype=float))
    out = []
    for f, c in zip(fracs.tolist(), np.asarray(Xc, dtype=int).tolist()):
        vals = {n: lo + f_i * (hi - lo) for (n, lo, hi), f_i in zip(DOMAIN["num"], f)}
        vals["gender"] = DOMAIN["cat"]["gender"][c[1] % 2]
        vals["smoker"] = DOMAIN["cat"]["smoker"][c[0] % 2]
        out.append(vals)
    return out


def to_csv(rows):
    head = "age,gender,smoker,blood_pressure,cholesterol"
    lines = [f"{r['age']:.1f},{r['gender']},{r['smoker']},{r['blood_pressure']:.1f},{r['cholesterol']:.1f}"
             for r in rows]
    return head + "\n" + "\n".join(lines)


def qwen_write(tok, mdl, csv_text, n_rows):
    msgs = [{"role": "system", "content": "You rewrite table rows as single sentences. "
             "Mention every value exactly as given. Reply with exactly the requested "
             "lines as 'ROW i: ...' and add no new facts."},
            {"role": "user", "content": f"Rewrite these {n_rows} patient rows (CSV):\n{csv_text}\n"
                                        f"Reply with exactly {n_rows} lines."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(ids, max_new_tokens=80 * n_rows, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:])


def verify(rows, text):
    lines = [ln for ln in text.strip().split("\n") if ln.startswith("ROW")]
    hits, total = 0, 0
    for i, r in enumerate(rows):
        ln = lines[i].lower() if i < len(lines) else ""
        checks = {f"{k}": (str(int(v)) in ln if isinstance(v, float) else
                           (v in ln and (v in ("female", "non-smoker") or
                            ("female" not in ln if v == "male" else "non-smoker" not in ln))))
                  for k, v in r.items()}
        hits += sum(checks.values())
        total += len(checks)
        print(f"row {i + 1}: " + " ".join(f"{k}={'OK' if v else 'MISS'}" for k, v in checks.items()))
    print(f"slot fidelity: {hits}/{total} = {hits / total:.3f}")
    return hits / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    Xn = rng.randn(args.rows, 3)
    Xc = rng.randint(0, 2, size=(args.rows, 2))
    rows = display_rows(Xn, Xc)
    csv_text = to_csv(rows)
    print(csv_text, "\n----")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="float16").cuda().eval()
    out = qwen_write(tok, mdl, csv_text, args.rows)
    print(out)
    verify(rows, out)


if __name__ == "__main__":
    main()
