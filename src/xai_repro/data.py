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


def load_wikitext103(seq_len: int = 256) -> LMDataset:
    """Load and tokenize WikiText-103 into fixed-length LM blocks.

    Parameters
    ----------
    seq_len:
        Block length in tokens. The reproduction uses 256.
    """
    cache = _cache_dir()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=str(cache))
    tokenizer.pad_token = tokenizer.eos_token

    raw: DatasetDict = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=str(cache))

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="tokenize",
        num_proc=4,
    )

    def group(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        # Concatenate then chunk into blocks of ``seq_len``.
        concatenated: list[int] = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total = (len(concatenated) // seq_len) * seq_len
        blocks = [concatenated[i : i + seq_len] for i in range(0, total, seq_len)]
        return {"input_ids": blocks, "labels": [list(b) for b in blocks]}

    chunked = tokenized.map(
        group,
        batched=True,
        batch_size=1000,
        num_proc=4,
        desc=f"chunk into {seq_len}-token blocks",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return LMDataset(
        train=chunked["train"],
        validation=chunked["validation"],
        tokenizer=tokenizer,
        collator=collator,
    )


__all__ = ["LMDataset", "load_wikitext103"]
