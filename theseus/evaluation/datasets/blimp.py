"""
BLiMP (Benchmark of Linguistic Minimal Pairs) evaluation.

Tests grammatical knowledge via minimal pair acceptability judgments.
"""

import random
from dataclasses import dataclass
from typing import Tuple

import datasets
from datasets import load_dataset, get_dataset_config_names

from theseus.config import configure, field
from theseus.evaluation.base import PerplexityComparisonEvaluation
from theseus.registry import evaluation


@dataclass
class BlimpConfig:
    subset: str = field("eval/blimp/subset", default="")


@evaluation("blimp")
class Blimp(PerplexityComparisonEvaluation):
    """BLiMP evaluation using perplexity comparison.

    Each sample contains a grammatically correct and incorrect sentence.
    The model should assign lower perplexity to the correct sentence.
    """

    CONFIG = BlimpConfig

    def __init__(self) -> None:
        subset = configure(self.CONFIG).subset or None
        if subset is not None:
            self.ds = load_dataset("nyu-mll/blimp", subset)["train"]
        else:
            configs = get_dataset_config_names("nyu-mll/blimp")
            all_ds = [load_dataset("nyu-mll/blimp", s) for s in configs]
            self.ds = datasets.concatenate_datasets([d["train"] for d in all_ds])
        self.R = random.Random(7)

    @property
    def name(self) -> str:
        return "blimp"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> Tuple[str, list[str], int]:
        """Get sample at index.

        Returns:
            (prefix, list_of_continuations, correct_index)
        """
        rev = self.R.choice([True, False])
        sample = self.ds[indx]
        continuations = [sample["sentence_good"], sample["sentence_bad"]]

        if rev:
            return ("", list(reversed(continuations)), 1)
        else:
            return ("", continuations, 0)
