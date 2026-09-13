"""
PG-19 (Gutenberg) perplexity evaluation.

Measures language model perplexity on Gutenberg books.
Returns 1/perplexity (higher is better).
"""

from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.evaluation.base import PerplexityEvaluation
from theseus.registry import evaluation


@dataclass
class PG19EvalConfig:
    num_samples: int = field("eval/pg19_ppl/num_samples", default=100)


@evaluation("pg19_ppl")
class PG19Eval(PerplexityEvaluation):
    """Perplexity evaluation on Project Gutenberg books."""

    CONFIG = PG19EvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
        self.items: list[str] = []
        for item in ds:
            text = item.get("TEXT", "")
            if text:
                self.items.append(text)
            if len(self.items) >= num_samples:
                break

    @property
    def name(self) -> str:
        return "pg19_ppl"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]
