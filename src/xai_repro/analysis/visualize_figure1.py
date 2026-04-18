#!/usr/bin/env python3
# Standard library
import gc
import os
import warnings
warnings.filterwarnings("ignore")

# Third-party
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import kurtosis as scipy_kurtosis
from transformers import AutoTokenizer, AutoModelForCausalLM

matplotlib.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
})

#
# CONFIGURATION
#
MODELS = [
    # (display_name,              hf_model_id,                           use_fp16  load_4bit)
    # -- Pythia series ---
    ("Pythia-14M",                "EleutherAI/pythia-14m",               False,    False),
    ("Pythia-31M",                "EleutherAI/pythia-31m",               False,    False),
    ("Pythia-70M",                "EleutherAI/pythia-70m",               False,    False),
    ("Pythia-160M",               "EleutherAI/pythia-160m",              False,    False),
    ("Pythia-410M",               "EleutherAI/pythia-410m",              False,    False),
    ("Pythia-1B",                 "EleutherAI/pythia-1b",                True,     False),
    ("Pythia-1.4B",               "EleutherAI/pythia-1.4b",              True,     False),
    # -- GPT-2 series ---
    ("GPT-2 (124M)",              "openai-community/gpt2",               False,    False),
    ("GPT-2-Medium (355M)",       "openai-community/gpt2-medium",        False,    False),
    ("GPT-2-Large (774M)",        "openai-community/gpt2-large",         False,    False),
    ("GPT-2-XL (1.5B)",           "openai-community/gpt2-xl",            True,     False),
    # -- Llama series ---
    # 8B: fits in fp16 on a 40 GB A100 (~16 GB)
    ("Llama-3-8B",                "NousResearch/Meta-Llama-3-8B",        True,     False),
]

# Sequence length fed to every model
SEQ_LEN: int = 512

# How many hidden channels to display in the activation heatmap
N_VIS_CHANNELS: int = 512

# Output figure path
SAVE_PATH: str = "figure1_replication.png"

# WikiText-103 validation set - opening passage (canonical evaluation text used in
# many attention / LLM papers, including Kaul et al. 2024).
# Source: https://huggingface.co/datasets/Salesforce/wikitext  split="validation"
_BASE_TEXT: str = (
    " = Valkyria Chronicles III = \n"
    " Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , "
    "lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles "
    "III outside Japan , is a tactical role @-@ playing game developed by Sega and "
    "Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it "
    "is the third game in the Valkyria series . Employing the same fusion of tactical "
    "and real @-@ time gameplay as its predecessors , the story runs parallel to the "
    "first game and follows the \" Nameless \" , a penal military unit serving the nation "
    "of Gallia during the Second Europan War who perform secret black operations and are "
    "pitted against the Imperial unit \" Calamity Raven \" . The game began development "
    "in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . "
    "While it retained the standard features of the series , it also underwent multiple "
    "adjustments , such as making the game more forgiving for series newcomers . Character "
    "designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous "
    "entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large number "
    "of assets from Valkyria Chronicles II were carried over to its successor . After its "
    "release , Valkyria Chronicles III received mixed reviews . It was praised for its "
    "pacing and gameplay , but criticized for lack of innovation . It sold 165 @,@ 077 "
    "units within its first week and 339 @,@ 186 units in total . A port for mobile "
    "platforms was released in 2012 . \n"
    " = = Gameplay = = \n"
    " Like previous Valkyria Chronicles games , Valkyria Chronicles III is a "
    "tactical role @-@ playing game where players take control of a military unit and "
    "take turns with the enemy in completing objectives . The game features a system "
    "called BLiTZ ( Battle of Live Tactical Zones ) inherited from previous games in "
    "the series . During their turn the player views an overhead map in Command Mode , "
    "and then adjusts the positioning of their characters . Characters and units each "
    "have various statistics determining their performance in combat . "
)

#
# INPUT PREPARATION
#
def build_input_ids(tokenizer, seq_len: int, device: str) -> torch.Tensor:
    """
    Tokenise _BASE_TEXT, repeating it until we have at least seq_len tokens,
    then truncate to exactly seq_len.  Returns shape (1, seq_len).
    """
    base_ids = tokenizer.encode(_BASE_TEXT, add_special_tokens=True)
    ids = base_ids[:]
    while len(ids) < seq_len:
        ids += tokenizer.encode(_BASE_TEXT, add_special_tokens=False)
    ids = ids[:seq_len]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

#
# XAI METRICS
#
def token0_attention_percent(attn: np.ndarray) -> float:
    dominant_key = np.argmax(attn, axis=1)
    return 100.0 * float((dominant_key == 0).mean())


# Attention-sink detection
def _expected_causal_col_mean(L: int) -> np.ndarray:
    """
    For each key position k ∈ [0, L-1], the expected mean attention weight
    received under a perfectly *uniform* causal model - i.e., if every query q
    distributes its attention equally over the q+1 tokens it can see.

        E[col_mean[k]] = (1/L) · Σ_{q=k}^{L-1}  1/(q+1)
                       = (1/L) · (H_L - H_k)

    where H_n = Σ_{i=1}^n 1/i  is the n-th harmonic number.

    This is the causal-position baseline.  Key-0 naturally has a high
    col_mean because *all* queries can attend to it; key-511 has a tiny
    baseline because only query-511 can attend to it.  Dividing the
    observed col_mean by this baseline yields an *excess ratio* that is
    1.0 for uniform attention and > 1.0 for a true sink.
    """
    H = np.zeros(L + 1)
    H[1:] = np.cumsum(1.0 / np.arange(1, L + 1))   # H[k] = Σ_{i=1}^k 1/i
    k_arr = np.arange(L)
    return (H[L] - H[k_arr]) / L                     # shape (L,)


def identify_sinks(
    attn_mean:    np.ndarray,
    dom_pct:      np.ndarray,
    z_thresh:     float = 3.0,
    min_dom_pct:  float = 2.0,
) -> list[dict]:
    """
    Identify attention-sink positions using two complementary criteria (OR).

    Parameters
    ---
    attn_mean : (L, L) - attention averaged over ALL layers + heads.
                Used only for the causal-excess z-score (structural criterion).

    dom_pct   : (L,) - **paper-faithful metric**.
                For every (layer, head, query) triple, we record which key k
                had the highest attention weight.  dom_pct[k] is the percentage
                of those triples where key k won.  This exactly matches the
                paper's "% of (query, head) pairs" figure (we additionally
                average over layers).

                NOTE: This is NOT computed from argmax(attn_mean) - that would
                inflate values to ~100 % for all models because averaging
                layers+heads makes column-0 win every row of the mean matrix.

    Criteria
    ---
    1. dom_pct[k] ≥ min_dom_pct      - token k is a genuine attentional focus
    2. log-excess z-score ≥ z_thresh  - column k is a structural outlier
                                        above the causal-position baseline
    """
    L = attn_mean.shape[0]

    col_mean = attn_mean.mean(axis=0)                   # (L,)
    baseline = _expected_causal_col_mean(L)             # (L,)
    excess   = col_mean / (baseline + 1e-12)            # (L,)

    log_excess = np.log(excess + 1e-12)
    mu, sigma  = log_excess.mean(), log_excess.std()
    z_score    = (log_excess - mu) / (sigma + 1e-10)    # (L,)

    is_sink = (dom_pct >= min_dom_pct) | (z_score >= z_thresh)

    sinks = [
        {
            "pos":      int(k),
            "col_mean": float(col_mean[k]),
            "dom_pct":  float(dom_pct[k]),
            "excess":   float(excess[k]),
            "z_score":  float(z_score[k]),
        }
        for k in np.where(is_sink)[0]
    ]
    sinks.sort(key=lambda s: -s["dom_pct"])
    return sinks


def max_token_kurtosis(hidden: np.ndarray) -> float:
    """
    Parameters
    ---
    hidden : (L, D) float32 - last-layer hidden states.

    Returns
    ---
    Maximum Pearson kurtosis over all token positions.
    (Gaussian baseline = 3; values >> 3 indicate outlier channels.)

    This mirrors Eq. (2) in the paper:
        κ_{m,l} = E_d[(X - μ)^4] / (E_d[(X - μ)^2])^2
    """
    L = hidden.shape[0]
    best = 3.0
    for t in range(L):
        x = hidden[t].astype(np.float64)
        if x.std() < 1e-8:
            continue
        k = float(scipy_kurtosis(x, fisher=False))   # fisher=False -> Gaussian = 3
        if k > best:
            best = k
    return best

#
# MODEL LOADING & FEATURE EXTRACTION
#
def _first_device(model) -> str:
    """Return the device string of the first parameter (works for device_map='auto')."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda"


def extract_features(
    name:        str,
    model_id:    str,
    use_fp16:    bool,
    load_4bit:   bool = False,
) -> tuple[np.ndarray, np.ndarray, float, list]:
    """
    Load *model_id*, forward a 512-token prompt, and return:

      attn_mean : (L, L)                      - mean attention over ALL layers+heads
      act_abs   : (L, min(D, N_VIS_CHANNELS)) - |last-layer hidden state|
      kurt      : float                       - max per-token kurtosis
      sinks     : list[dict]                  - identified attention-sink positions

    Parameters
    ---
    use_fp16   : load in float16 with device_map="auto"  (models ≥ ~1 B params)
    load_4bit  : load in NF4 4-bit quantisation via bitsandbytes (models ≥ ~30 B)
                 Requires:  pip install bitsandbytes
                 Overrides use_fp16 - the model is loaded at 4-bit regardless.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if (use_fp16 or load_4bit) else torch.float32

    _banner(name, model_id)

    # -- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Model ---
    # attn_implementation="eager" must be set for ALL models, not just large ones.
    # Since transformers >= 4.36, the default changed to "sdpa" (PyTorch fused
    # kernel) which does NOT materialise the L×L attention matrix ->
    # outputs.attentions returns a tuple of None, causing TypeError downstream.
    load_kw: dict = dict(torch_dtype=dtype, attn_implementation="eager")

    if load_4bit:
        # 4-bit NF4 quantisation: the model weights are stored in ~4 bits but
        # all compute (including attention) runs in float16.
        # Llama-3-70B: ~140 GB fp16 -> ~35 GB in 4-bit - fits on a single A100.
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   # saves ~0.4 bits/param extra
        )
        load_kw["device_map"] = "auto"
        load_kw.pop("torch_dtype", None)      # BitsAndBytesConfig sets dtype internally
    elif use_fp16:
        load_kw["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    if not (use_fp16 or load_4bit):
        model = model.to(device)
    model.eval()

    # -- Build input ---
    input_device = _first_device(model)
    input_ids = build_input_ids(tokenizer, SEQ_LEN, input_device)

    # -- Forward pass ---
    # output_attentions=True  -> outputs.attentions  : tuple[(batch, heads, L, L)]
    # output_hidden_states=True -> outputs.hidden_states : tuple[(batch, L, D)]
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            output_hidden_states=True,
        )

    # -- Attention - average over ALL layers AND all heads (paper §Figure 1) --
    # outputs.attentions : tuple of n_layers tensors, each (1, n_heads, L, L)
    # We compute a running sum layer-by-layer to avoid holding every layer
    # in RAM simultaneously (important for Llama-3-8B: 32 layers × 32 heads).
    if outputs.attentions is None:
        raise RuntimeError(
            f"Model {model_id} returned outputs.attentions=None. "
            "Try passing attn_implementation='eager'."
        )
    n_layers   = len(outputs.attentions)
    attn_acc   = None                                    # accumulator (L, L)
    dom_counts = None                                    # (L,) int64 - per-head argmax tally
    total_qh   = 0                                       # total (query, head, layer) triples
    for layer_attn in outputs.attentions:               # (1, n_heads, L, L)
        layer_np = layer_attn[0].float().cpu().numpy()  # (n_heads, L, L)
        layer_mean = layer_np.mean(axis=0)              # (L, L) - mean over heads
        attn_acc   = layer_mean if attn_acc is None else attn_acc + layer_mean
        # Paper-faithful dom_pct: for each (head, query) triple record which key wins
        max_keys = layer_np.argmax(axis=2)              # (n_heads, L) - winning key per query
        n_h, L_  = max_keys.shape
        if dom_counts is None:
            dom_counts = np.zeros(L_, dtype=np.int64)
        for h_idx in range(n_h):
            dom_counts += np.bincount(max_keys[h_idx], minlength=L_)
        total_qh += n_h * L_
    attn_mean       = attn_acc / n_layers               # (L, L) - mean over layers+heads
    dom_pct_perhead = 100.0 * dom_counts / total_qh     # (L,) - % of triples each key wins

    # -- Hidden states - LAST layer only (paper Figure 1) ---
    # outputs.hidden_states : tuple of (n_layers+1) tensors, each (1, L, D)
    # Index 0 = raw token embedding; index -1 = final transformer block output.
    # The paper visualises the last-layer hidden states for activation outliers.
    hidden_np = outputs.hidden_states[-1][0].float().cpu().numpy()  # (L, D)

    # -- XAI metrics ---
    kurt  = max_token_kurtosis(hidden_np)
    sinks = identify_sinks(attn_mean, dom_pct_perhead)

    print(f"  Max Token Kurtosis  : {kurt:>10.1f}")
    if sinks:
        print(f"  Attention sinks detected ({len(sinks)}):")
        print(f"    {'pos':>5}  {'dom_pct':>8}  {'excess':>8}  {'z_score':>8}")
        for s in sinks:
            print(f"    {s['pos']:>5}  {s['dom_pct']:>7.2f}%  "
                  f"{s['excess']:>7.1f}×  {s['z_score']:>8.2f}")
    else:
        print("  No dominant attention sinks detected.")

    # -- Activation slice for visualisation ---
    n_ch    = min(N_VIS_CHANNELS, hidden_np.shape[1])
    act_abs = np.abs(hidden_np[:, :n_ch])               # (L, n_ch)

    # -- GPU cleanup ---
    # Note: last_attn_tensor / last_hidden_tensor no longer exist - the
    # all-layers averaging loop replaced them with attn_acc / hidden_acc.
    del model, outputs, attn_acc, input_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return attn_mean, act_abs, kurt, sinks


def _banner(name: str, model_id: str) -> None:
    print(f"\n{'-'*64}")
    print(f"  {name}  |  {model_id}")
    print(f"{'-'*64}")

#
# PLOTTING HELPERS
#
_CMAP = "Blues"  # sequential blue colormap, matching the paper's palette


def _attach_colorbar(fig: plt.Figure, ax: plt.Axes, im) -> None:
    """Append a slim vertical colorbar to the right of *ax* without resizing it."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)


def _sparse_ticks(length: int, n_ticks: int = 5) -> list[int]:
    """Return n_ticks evenly-spaced tick positions for an axis of *length*."""
    step = max(1, length // (n_ticks - 1))
    ticks = list(range(0, length, step))
    if ticks[-1] != length - 1:
        ticks.append(length - 1)
    return ticks


def plot_model_row(
    fig:       plt.Figure,
    ax_attn:   plt.Axes,
    ax_act:    plt.Axes,
    attn_mean: np.ndarray,        # (L, L)
    act_abs:   np.ndarray,        # (L, C)
    name:      str,
    kurt:      float,
    sinks:     list[dict],        # output of identify_sinks()
) -> None:
    """
    Render the two heatmaps for one model row, replicating Figure 1 style:

    Left  - Attention matrix  (Query × Key).
              Coloured boxes mark every detected sink column (not only k=0).
    Right - Activation magnitude  (Token × Channel).
              Red vertical lines mark the top outlier channels.
    """
    L = attn_mean.shape[0]
    C = act_abs.shape[1]

    attn_ticks = _sparse_ticks(L)
    ch_ticks   = [0, C // 4, C // 2, 3 * C // 4, C - 1]

    # -- Left: Attention heatmap ---
    # vmax = raw maximum of the attention matrix (no clipping).
    # attn[0,0] = 1.0 because causal masking forces token-0 to attend only to
    # itself.  The resulting faint diagonal is NOT a visualisation bug - it is
    # the attention-sink phenomenon: token-0 monopolises attention weight, so
    # self-attention on the diagonal is genuinely small.  Do not rescale.
    im_attn = ax_attn.imshow(
        attn_mean,
        cmap=_CMAP,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=attn_mean.max(),
    )

    # -- Title: list all detected sinks ---
    if sinks:
        parts = [
            f"k={s['pos']} ({s['dom_pct']:.1f}%)"
            for s in sinks[:3]
        ]
        extra = f" +{len(sinks)-3} more" if len(sinks) > 3 else ""
        title_line2 = "Sinks: " + ", ".join(parts) + extra
    else:
        title_line2 = "No dominant sinks"
    ax_attn.set_title(
        f"{name}\n{title_line2}",
        fontsize=8.5, fontweight="bold", pad=4,
    )

    ax_attn.set_xlabel("Key Position",   fontsize=7, labelpad=2)
    ax_attn.set_ylabel("Query Position", fontsize=7, labelpad=2)
    ax_attn.set_xticks(attn_ticks)
    ax_attn.set_xticklabels([str(t) for t in attn_ticks], fontsize=6)
    ax_attn.set_yticks(attn_ticks)
    ax_attn.set_yticklabels([str(t) for t in attn_ticks], fontsize=6)
    ax_attn.tick_params(length=2)

    # Shift x-axis left by a small margin so column-0 is not flush against the
    # y-axis spine.  Without this, the sink bar at k=0 merges visually with the
    # plot border and is hard to distinguish.  The right limit is kept at the
    # natural image boundary (L - 0.5).
    _X_PAD = max(3, int(L * 0.01))   # 1 % of sequence length, minimum 3 units
    ax_attn.set_xlim(-_X_PAD, L - 0.5)

    # Sink positions are reported in the title only - no overlays drawn on the
    # heatmap so the original blue gradient is fully preserved.
    _attach_colorbar(fig, ax_attn, im_attn)

    # -- Right: Activation magnitude heatmap ---
    # Clip the color scale at the 99th percentile so extreme outliers
    # do not wash out the rest - but the outliers are still visible.
    vmax_act = float(np.percentile(act_abs, 99.5))
    im_act = ax_act.imshow(
        act_abs,
        cmap=_CMAP,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax_act,
    )
    ax_act.set_title(
        f"Activation Magnitude  |x|\n"
        r"Max Kurtosis: " + f"{kurt:.0f}",
        fontsize=9, fontweight="bold", pad=4,
    )
    ax_act.set_xlabel("Channel Index", fontsize=7, labelpad=2)
    ax_act.set_ylabel("Token Index",   fontsize=7, labelpad=2)
    ax_act.set_xticks(ch_ticks)
    ax_act.set_xticklabels([str(c) for c in ch_ticks], fontsize=6)
    ax_act.set_yticks(attn_ticks)
    ax_act.set_yticklabels([str(t) for t in attn_ticks], fontsize=6)
    ax_act.tick_params(length=2)

    # Mark the top-5 "outlier" channels with a light red vertical line.
    # A channel is an outlier if its max |activation| across tokens is
    # more than 3× the per-channel mean.
    col_max  = act_abs.max(axis=0)                        # (C,)
    col_mean = act_abs.mean(axis=0).mean()
    outlier_cols = np.where(col_max > 3.0 * col_mean)[0]
    # Sort by magnitude and keep the top 5 for clarity
    outlier_cols = sorted(outlier_cols, key=lambda c: -col_max[c])[:5]
    for c in outlier_cols:
        ax_act.axvline(x=c, color="red", linewidth=0.8, alpha=0.7, zorder=5)

    # Circle the first-token row to match the paper (paper circles token 0)
    ax_act.add_patch(Rectangle(
        (-0.5, -0.5), float(C), 1.0,
        linewidth=1.5, edgecolor="red", facecolor="none", zorder=5,
    ))

    _attach_colorbar(fig, ax_act, im_act)


def _plot_error_row(axes_row, name: str, exc: Exception) -> None:
    """Fill both axes with a red error card when model loading fails."""
    msg = (
        f"Failed to load  {name}\n\n"
        f"{type(exc).__name__}:\n{str(exc)[:200]}"
    )
    for ax in axes_row:
        ax.set_facecolor("#fff5f5")
        ax.text(
            0.5, 0.5, msg,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=7.5, color="crimson", wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="crimson", lw=1),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=9, fontweight="bold", color="crimson")

#
# MAIN
#
def main() -> None:
    n_models = len(MODELS)

    # Each model occupies one row; two columns (attention | activation).
    fig, axes = plt.subplots(
        nrows=n_models,
        ncols=2,
        figsize=(13.5, 3.5 * n_models),
        gridspec_kw={"wspace": 0.55, "hspace": 0.80},
    )
    # Guarantee 2-D indexing even for a single model
    if n_models == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Attention Sinks & Activation Outliers in LLMs\n"
        "Replication of Figure 1 - "
        '"From Attention to Activation" (Kaul et al., arXiv:2410.17174)',
        fontsize=12, fontweight="bold", y=1.003,
    )

    summary: list[tuple] = []

    for idx, (name, model_id, use_fp16, load_4bit) in enumerate(MODELS):
        try:
            attn_mean, act_abs, kurt, sinks = extract_features(
                name, model_id, use_fp16, load_4bit
            )
            plot_model_row(
                fig,
                axes[idx, 0], axes[idx, 1],
                attn_mean, act_abs,
                name, kurt, sinks,
            )
            summary.append((name, kurt, sinks, True))

        except Exception as exc:
            print(f"\n[ERROR] {name}: {type(exc).__name__}: {exc}")
            _plot_error_row(axes[idx], name, exc)
            summary.append((name, None, [], False))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -- Save ---
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    print(f"\n{'-'*64}")
    print(f"  Figure saved  ->  {SAVE_PATH}")
    print(f"{'-'*64}")

    # -- Console summary: per-model sink table ---
    sep = "-"
    print(f"\n{'Model':<26} {'Kurt':>8}  "
          f"{'Sink pos':>8}  {'dom%':>7}  {'excess':>8}  {'z':>6}")
    print(sep * 70)
    for name, kurt, sinks, ok in summary:
        if not ok:
            print(f"{name:<26} {'ERROR':>8}")
            continue
        kurt_s = f"{kurt:>8.1f}"
        if not sinks:
            print(f"{name:<26} {kurt_s}  {'(none)':>8}")
        else:
            for i, s in enumerate(sinks):
                label = name if i == 0 else ""
                ks    = f"{kurt_s if i == 0 else '':>8}"
                print(f"{label:<26} {ks}  "
                      f"{s['pos']:>8}  {s['dom_pct']:>6.2f}%  "
                      f"{s['excess']:>7.1f}×  {s['z_score']:>6.2f}")
    print(sep * 70)
    print()

    plt.show()


if __name__ == "__main__":
    main()
