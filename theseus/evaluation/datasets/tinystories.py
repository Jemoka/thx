"""
TinyStories perplexity evaluation.

Measures language model perplexity on the TinyStories validation set.
Returns 1/perplexity (higher is better).
"""

from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.evaluation.base import PerplexityEvaluation
from theseus.registry import evaluation


@dataclass
class TinyStoriesEvalConfig:
    num_samples: int = field("eval/tinystories_ppl/num_samples", default=500)


@evaluation("tinystories_ppl")
class TinyStoriesEval(PerplexityEvaluation):
    """Perplexity evaluation on TinyStories validation stories.

    Loads from roneneldan/TinyStories on HuggingFace.
    """

    CONFIG = TinyStoriesEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset("roneneldan/TinyStories", split="validation")
        self.ds = ds.select(range(min(num_samples, len(ds))))

    @property
    def name(self) -> str:
        return "tinystories_ppl"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> str:
        return self.ds[indx]["text"]  # type: ignore
