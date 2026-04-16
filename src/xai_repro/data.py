"""WikiText-103 tokenization pipeline for the 60M GPT-2 reproduction.

The raw corpus is concatenated and chunked into fixed-length blocks of
``seq_len`` tokens (no sentence boundaries, no padding). The tokenized
arrow shards are cached under ``$SCRATCH/hf_cache`` (or
``~/.cache/huggingface`` if ``$SCRATCH`` is unset) so the three
experimental runs share a single tokenization pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers import DataCollatorForLanguageModeling, GPT2TokenizerFast


def _cache_dir() -> Path:
    scratch = os.environ.get("SCRATCH")
    base = Path(scratch) if scratch else Path.home() / ".cache"
    cache = base / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@dataclass
class LMDataset:
    """Container for the tokenized LM splits and the collator."""

    train: Dataset
    validation: Dataset
    tokenizer: GPT2TokenizerFast
    collator: DataCollatorForLanguageModeling


def load_c4(seq_len: int = 256) -> LMDataset:
    """Load and tokenize C4 (en) into fixed-length LM blocks via streaming.

    Parameters
    ----------
    seq_len:
        Block length in tokens. The reproduction uses 256.
    """
    cache = _cache_dir()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=str(cache))
    tokenizer.pad_token = tokenizer.eos_token

    # Use streaming=True to avoid multi-terabyte downloads on the cluster
    ds = load_dataset("allenai/c4", "en", streaming=True, cache_dir=str(cache))
    
    # Take a fixed validation set for faster eval (streaming doesn't support easy length)
    # The paper doesn't specify val size, 5000 examples is usually plenty for P100.
    # Paper §5 evaluates on 100 000 validation samples — required for stable
    # tail statistics (kurtosis, max|activation|) on heavy-tailed distributions.
    val_subset = ds["validation"].take(100_000)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], add_special_tokens=False)

    def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        # Concatenate then chunk into blocks of ``seq_len``.
        concatenated: list[int] = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total = (len(concatenated) // seq_len) * seq_len
        if total == 0:
            return {"input_ids": [], "labels": []}
        blocks = [concatenated[i : i + seq_len] for i in range(0, total, seq_len)]
        return {"input_ids": blocks, "labels": [list(b) for b in blocks]}

    # C4 ships with {text, timestamp, url}; strip all non-token columns so the
    # collator (which calls `tokenizer.pad`) doesn't try to tensorize strings.
    c4_drop_cols = ["text", "timestamp", "url"]
    tokenized_train = ds["train"].map(tokenize, batched=True, remove_columns=c4_drop_cols)
    tokenized_val = val_subset.map(tokenize, batched=True, remove_columns=c4_drop_cols)

    # Drop attention_mask — not needed for fixed-length causal LM blocks
    tokenized_train = tokenized_train.remove_columns("attention_mask")
    tokenized_val = tokenized_val.remove_columns("attention_mask")

    chunked_train = tokenized_train.map(group_texts, batched=True, batch_size=1000)
    chunked_val = tokenized_val.map(group_texts, batched=True, batch_size=1000)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return LMDataset(
        train=chunked_train,
        validation=chunked_val,
        tokenizer=tokenizer,
        collator=collator,
    )


__all__ = ["LMDataset", "load_c4"]
