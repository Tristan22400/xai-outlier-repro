#!/usr/bin/env python3
"""Input-only ablations for attention sink - NO retraining required.

All experiments run a single forward pass on a pretrained model:

    baseline    : natural input, BOS policy default.
    remove_bos  : drop the <bos> token (Llama-style models only).
    prepend     : prepend a rare/neutral token; check if sink follows position 0.
    length      : sweep sequence length; see how sink scales.
    zero_pe     : zero out learned position embeddings (GPT-2 style only);
                  PSEUDO-ablation - model degraded, interpret with care.

Metrics (Kaul et al. 2024 §2.1):
    A  - argmax %: fraction of (layer, head, query) triples whose argmax
         falls on the probed key position.
    B  - mass   %: mean attention weight on the probed key position.

Usage
---
  python input_ablation.py --models openai-community/gpt2-medium
  python input_ablation.py --models NousResearch/Meta-Llama-3-8B --fp16
  python input_ablation.py --models gpt2 gpt2-medium --experiments baseline prepend length
"""
from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# shared natural-text prompt (~600 tokens of English)
TEXT = (
    "A language model is a probability distribution over sequences of words. "
    "Given any sequence of words, a language model assigns a probability to the whole sequence. "
    "Modern neural language models, and in particular Transformer-based autoregressive models, "
    "achieve strong performance by conditioning each next-token prediction on all previous tokens. "
    "Attention mechanisms compute, for every query position, a weighted combination of all previous key positions, "
    "with weights produced by a softmax over dot-product scores. Because of causal masking, "
    "the first position in the sequence is the only key that every query can attend to. "
    "Recent work identifies a striking regularity: a disproportionate share of attention mass concentrates on this first key, "
    "regardless of its semantic content. This phenomenon is known as the attention sink. "
    "At the same time, hidden-state activations develop extreme outliers in specific feature channels. "
    "Together these two behaviours complicate the quantisation of large language models and motivate "
    "architectural or optimisation changes. The first token is privileged not by any intrinsic meaning "
    "but purely by its position; every downstream query must allocate some probability mass among its available keys, "
    "and the first key is always available. Practical consequences range from streaming inference stability to "
    "calibration of post-training quantisation. In this script we probe, with input manipulations alone, "
    "whether sink behaviour follows position or token identity, whether it scales with context length, "
    "and whether it survives the removal of common architectural choices such as positional embeddings. "
    "The experiments require no retraining, only one forward pass per configuration on a pretrained model. "
    "Outputs are the two metrics used in Kaul et al. 2024 Section 2.1: the fraction of (layer, head, query) "
    "triples whose argmax falls on the probed key, and the mean attention weight assigned to that key. "
    "Additional context is added here to pad the prompt to a useful length for the 512-token default, "
    "including references to tokenisation, normalisation, and feed-forward non-linearity, which are not "
    "the primary variables under study but provide realistic distributional input to the attention mechanism."
) * 3

OUT = Path("ablation_results")
OUT.mkdir(exist_ok=True)


# utilities
def first_device(model) -> str:
    return str(next(model.parameters()).device)


def load(model_id: str, fp16: bool = False):
    dtype = torch.float16 if fp16 else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(torch_dtype=dtype, attn_implementation="eager")
    if fp16:
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not fp16:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tok


def tokenize(tok, text: str, seq_len: int, device: str, add_bos: bool = True):
    ids = tok(text, return_tensors="pt", add_special_tokens=add_bos).input_ids
    if ids.size(1) >= seq_len:
        ids = ids[:, :seq_len]
    else:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        pad = torch.full((1, seq_len - ids.size(1)), pad_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    return ids.to(device)


def forward_attn(model, ids):
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError("Attention tensors are None - ensure attn_implementation='eager'.")
    return torch.stack([a[0].float().cpu() for a in out.attentions])   # (L, H, T, T)


def sink_metric(attn, key_pos: int = 0):
    argmax_keys = attn.argmax(dim=-1)                                   # (L, H, T)
    argmax_pct  = (argmax_keys == key_pos).float().mean().item() * 100
    mass_pct    = attn[..., key_pos].mean().item() * 100
    return round(argmax_pct, 3), round(mass_pct, 3)


def max_seq(model) -> int:
    cfg = model.config
    return min(
        getattr(cfg, "max_position_embeddings", 10**9),
        getattr(cfg, "n_positions",             10**9),
    )


# experiments
def exp_baseline(model, tok, device, seq_len=512):
    ids  = tokenize(tok, TEXT, seq_len, device, add_bos=True)
    attn = forward_attn(model, ids)
    a, m = sink_metric(attn, 0)
    return {"experiment": "baseline", "seq_len": seq_len,
            "argmax_pct_key0": a, "mass_pct_key0": m,
            "first_token": tok.decode([int(ids[0, 0])])}


def exp_remove_bos(model, tok, device, seq_len=512):
    if tok.bos_token_id is None:
        return {"experiment": "remove_bos", "skipped": "tokenizer has no BOS"}
    ids  = tokenize(tok, TEXT, seq_len, device, add_bos=False)
    if int(ids[0, 0]) == tok.bos_token_id:
        return {"experiment": "remove_bos", "skipped": "BOS still appears at pos 0"}
    attn = forward_attn(model, ids)
    a, m = sink_metric(attn, 0)
    return {"experiment": "remove_bos", "seq_len": seq_len,
            "argmax_pct_key0": a, "mass_pct_key0": m,
            "first_token": tok.decode([int(ids[0, 0])]),
            "first_token_id": int(ids[0, 0])}


def exp_prepend(model, tok, device, seq_len=512):
    # Pick a 'neutral' token: mid-vocabulary, not special. Reproducible choice.
    prep_id = min(tok.vocab_size - 1, 5000)
    # Skip special-token range just in case
    if tok.all_special_ids and prep_id in tok.all_special_ids:
        prep_id = 1000
    ids_core = tokenize(tok, TEXT, seq_len - 1, device, add_bos=False)
    prep = torch.full((1, 1), prep_id, dtype=ids_core.dtype, device=device)
    ids  = torch.cat([prep, ids_core], dim=1)
    attn = forward_attn(model, ids)
    a0, m0 = sink_metric(attn, 0)      # new prepended token
    a1, m1 = sink_metric(attn, 1)      # what used to be position 0
    return {"experiment": "prepend", "seq_len": seq_len,
            "prepend_token":    tok.decode([prep_id]),
            "prepend_token_id": int(prep_id),
            "argmax_pct_key0":  a0, "mass_pct_key0":  m0,
            "argmax_pct_key1":  a1, "mass_pct_key1":  m1}


def exp_length_sweep(model, tok, device, lengths=(32, 128, 512, 1024, 2048)):
    cap = max_seq(model)
    rows = []
    for L in lengths:
        if L > cap:
            rows.append({"seq_len": L, "skipped": f"exceeds model max {cap}"})
            continue
        ids  = tokenize(tok, TEXT, L, device, add_bos=True)
        attn = forward_attn(model, ids)
        a, m = sink_metric(attn, 0)
        rows.append({"seq_len": L, "argmax_pct_key0": a, "mass_pct_key0": m})
    return {"experiment": "length_sweep", "rows": rows}


def exp_zero_pe(model, tok, device, seq_len=512):
    # Only GPT-2 style models expose learnable absolute PE as `transformer.wpe`.
    if not (hasattr(model, "transformer") and hasattr(model.transformer, "wpe")):
        return {"experiment": "zero_pe", "skipped": "no wpe module (not GPT-2 style)"}
    wpe   = model.transformer.wpe
    saved = wpe.weight.data.clone()
    wpe.weight.data.zero_()
    try:
        ids  = tokenize(tok, TEXT, seq_len, device, add_bos=True)
        attn = forward_attn(model, ids)
        a, m = sink_metric(attn, 0)
    finally:
        wpe.weight.data.copy_(saved)
    return {"experiment": "zero_pe", "seq_len": seq_len,
            "argmax_pct_key0": a, "mass_pct_key0": m,
            "note": "PSEUDO ablation - model trained WITH PE; PPL degrades. "
                    "If sink persists => learned PE pattern is not the cause. "
                    "If sink vanishes => inconclusive (model may simply be broken)."}


def exp_swap_at(model, tok, device, seq_len=512, swap_pos=0, window=5):
    """Replace a single token at ``swap_pos`` with a neutral id, keeping
    every other position identical. Compare mass/argmax distribution at the
    swap position and its neighbourhood before vs. after.

    If sink follows POSITION -> mass @ swap_pos should be preserved.
    If sink follows TOKEN identity -> mass @ swap_pos should collapse.
    """
    ids = tokenize(tok, TEXT, seq_len, device, add_bos=False)
    neutral_id = min(tok.vocab_size - 1, 5000)
    if tok.all_special_ids and neutral_id in tok.all_special_ids:
        neutral_id = 1000
    orig_tok_id  = int(ids[0, swap_pos])
    orig_tok_str = tok.decode([orig_tok_id])

    ids_swap = ids.clone()
    ids_swap[0, swap_pos] = neutral_id

    attn_base = forward_attn(model, ids)
    attn_swap = forward_attn(model, ids_swap)

    keys = sorted({0, 1} | {k for k in range(max(0, swap_pos - window),
                                              min(seq_len, swap_pos + window + 1))})
    rows = []
    for k in keys:
        ab, mb = sink_metric(attn_base, k)
        as_, ms_ = sink_metric(attn_swap, k)
        rows.append({"key": k,
                     "base_argmax": ab, "base_mass": mb,
                     "swap_argmax": as_, "swap_mass": ms_,
                     "delta_argmax": round(as_ - ab, 3),
                     "delta_mass":   round(ms_ - mb, 3)})
    return {"experiment":        f"swap@{swap_pos}",
            "swap_pos":           swap_pos,
            "original_token":     orig_tok_str,
            "original_token_id":  orig_tok_id,
            "neutral_token_id":   int(neutral_id),
            "rows":               rows}


def _make_swap_exp(pos: int):
    def _run(model, tok, device):
        return exp_swap_at(model, tok, device, swap_pos=pos)
    _run.__name__ = f"exp_swap_at_{pos}"
    return _run


EXPERIMENTS = {
    "baseline":   exp_baseline,
    "remove_bos": exp_remove_bos,
    "prepend":    exp_prepend,
    "length":     exp_length_sweep,
    "zero_pe":    exp_zero_pe,
    "swap_at_0":  _make_swap_exp(0),    # GPT-2 / GPT-2-M dominant position
    "swap_at_9":  _make_swap_exp(9),    # Pythia-160M dominant position
    "swap_at_50": _make_swap_exp(50),   # Pythia-31M dominant position
}


# orchestration
def run(model_id: str, fp16: bool, experiments):
    print(f"\n{'='*68}\n  {model_id}\n{'='*68}")
    model, tok = load(model_id, fp16)
    device = first_device(model)
    results = []
    for name in experiments:
        fn = EXPERIMENTS.get(name)
        if fn is None:
            print(f"  unknown experiment: {name}"); continue
        try:
            r = fn(model, tok, device)
        except Exception as e:
            r = {"experiment": name, "error": repr(e)}
        r["model"] = model_id
        results.append(r)
        print(f"  [{name:<11s}]  {json.dumps({k:v for k,v in r.items() if k not in ('model',)}, ensure_ascii=False)}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def print_summary(all_results):
    print("\n" + "="*92)
    print(f"  {'MODEL':<30}  {'EXPERIMENT':<12}  {'ARGMAX%':>8}  {'MASS%':>8}  notes")
    print("="*92)
    for r in all_results:
        model = r["model"].split("/")[-1]
        exp   = r["experiment"]
        if "skipped" in r:
            print(f"  {model:<30}  {exp:<12}  {'- skip -':>8}  {'':>8}  {r['skipped']}")
            continue
        if "error" in r:
            print(f"  {model:<30}  {exp:<12}  {'- err -':>8}  {'':>8}  {r['error'][:60]}")
            continue
        if exp == "length_sweep":
            for row in r["rows"]:
                L = row["seq_len"]
                if "skipped" in row:
                    label = f"L={L}"
                    print(f"  {model:<30}  {label:<12}  {'- skip -':>8}")
                else:
                    print(f"  {model:<30}  L={L:<10}  "
                          f"{row['argmax_pct_key0']:>7.2f}%  {row['mass_pct_key0']:>7.2f}%")
            continue
        if exp == "prepend":
            print(f"  {model:<30}  prepend key0  {r['argmax_pct_key0']:>7.2f}%  "
                  f"{r['mass_pct_key0']:>7.2f}%  <- new prepended token {r['prepend_token_id']}")
            print(f"  {model:<30}  prepend key1  {r['argmax_pct_key1']:>7.2f}%  "
                  f"{r['mass_pct_key1']:>7.2f}%  <- what used to be pos 0")
            continue
        if exp.startswith("swap@"):
            pos = r["swap_pos"]
            print(f"  {model:<30}  swap@{pos}  orig={r['original_token']!r}  ->  neutral id {r['neutral_token_id']}")
            print(f"    {'key':>5}  {'base arg%':>9}  {'base mass%':>10}  {'swap arg%':>9}  {'swap mass%':>10}  {'Δmass':>7}")
            for row in r["rows"]:
                marker = " <-" if row["key"] == pos else ""
                print(f"    {row['key']:>5}  {row['base_argmax']:>8.2f}%  {row['base_mass']:>9.2f}%  "
                      f"{row['swap_argmax']:>8.2f}%  {row['swap_mass']:>9.2f}%  {row['delta_mass']:>+7.2f}{marker}")
            continue
        print(f"  {model:<30}  {exp:<12}  {r['argmax_pct_key0']:>7.2f}%  "
              f"{r['mass_pct_key0']:>7.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["openai-community/gpt2-medium"],
                   help="HF model ids")
    p.add_argument("--fp16", action="store_true",
                   help="Load in float16 (required for ≥1B models)")
    p.add_argument("--experiments", nargs="+",
                   default=list(EXPERIMENTS.keys()),
                   help=f"Subset of {list(EXPERIMENTS)}")
    p.add_argument("--out", default=str(OUT / "results.json"))
    args = p.parse_args()

    all_results = []
    for mid in args.models:
        all_results.extend(run(mid, fp16=args.fp16, experiments=args.experiments))

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    json.dump(all_results, open(args.out, "w"), indent=2, ensure_ascii=False)
    print_summary(all_results)
    print(f"\nSaved -> {args.out}")
