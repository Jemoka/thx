"""
Pile perplexity evaluation.

Measures language model perplexity on a sample from the Pile.
Returns 1/perplexity (higher is better).
"""

from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.evaluation.base import PerplexityEvaluation
from theseus.registry import evaluation


@dataclass
class PileEvalConfig:
    num_samples: int = field("eval/pile_ppl/num_samples", default=500)


@evaluation("pile_ppl")
class PileEval(PerplexityEvaluation):
    """Perplexity evaluation on the Pile (EleutherAI)."""

    CONFIG = PileEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset(
            "parquet",
            data_files=(
                "hf://datasets/EleutherAI/pile@refs/convert/parquet/"
                "all/partial-test/*.parquet"
            ),
            split="train",
            streaming=True,
        )
        self.items: list[str] = []
        for item in ds:
            text = item.get("text", "")
            if text:
                self.items.append(text)
            if len(self.items) >= num_samples:
                break

    @property
    def name(self) -> str:
        return "pile_ppl"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]
