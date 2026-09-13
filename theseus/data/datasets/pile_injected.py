"""Pile with deterministic injected sequences for memorization evaluation.

Per Huang et al. (2024), injects verifiably-unknown token sequences into the
Pile at fixed positions so that retrieval can be measured after training.
"""

from collections.abc import Iterator

import numpy as np
from datasets import load_dataset

from theseus.data.datasets import StreamingPretrainingDataset
from theseus.registry import dataset

# Fixed vocabulary of "nonsense" words used to build injected sequences.
# These are deterministic gibberish strings that won't appear in natural text.
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _generate_injected_texts(
    n_sequences: int = 100,
    words_per_sequence: int = 80,
    seed: int = 42,
) -> list[str]:
    """Generate reproducible gibberish text sequences for injection.

    Each sequence is ~256 tokens worth of random "words" (3-8 chars each),
    designed to be verifiably absent from natural text.
    """
    rng = np.random.RandomState(seed)
    sequences: list[str] = []
    for _ in range(n_sequences):
        words = []
        for _ in range(words_per_sequence):
            length = rng.randint(3, 9)
            word = "".join(rng.choice(list(_ALPHABET), size=length))
            words.append(word)
        sequences.append(" ".join(words))
    return sequences


def _injection_positions(n_sequences: int, seed: int = 42) -> list[int]:
    """Deterministic document indices at which to inject sequences.

    Spread across the first ~10M documents with fixed gaps so that
    injection density is low enough to not distort pretraining.
    """
    rng = np.random.RandomState(seed + 1)
    gap = 100_000  # average gap between injections
    positions: list[int] = []
    pos = rng.randint(gap // 2, gap)
    for _ in range(n_sequences):
        positions.append(pos)
        pos += rng.randint(gap // 2, gap * 3 // 2)
    return positions


# Module-level constants for reproducibility across train and eval.
INJECTED_TEXTS = _generate_injected_texts()
INJECTION_POSITIONS = _injection_positions(len(INJECTED_TEXTS))


@dataset("pile_injected")
class PileInjected(StreamingPretrainingDataset):
    """The Pile with 100 deterministic injected sequences.

    Streams text from the Pile, inserting gibberish sequences at
    predetermined document indices. The injected texts are available
    as ``INJECTED_TEXTS`` for evaluation.
    """

    def __init__(self) -> None:
        self.ds = load_dataset(
            "parquet",
            data_files=(
                "hf://datasets/EleutherAI/pile@refs/convert/parquet/"
                "all/partial-train/*.parquet"
            ),
            split="train",
            streaming=True,
        )
        self._injection_set = dict(zip(INJECTION_POSITIONS, INJECTED_TEXTS))

    def __iter__(self) -> Iterator[str]:
        doc_idx = 0
        for item in self.ds:
            # Check if we need to inject before this document
            if doc_idx in self._injection_set:
                yield self._injection_set[doc_idx]
            text = item.get("text", "")
            if text:
                yield text
            doc_idx += 1
