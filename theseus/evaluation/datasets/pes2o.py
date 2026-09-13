"""
peS2o perplexity evaluation.

Measures language model perplexity on scientific papers from the peS2o corpus.
Returns 1/perplexity (higher is better).
"""

from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.evaluation.base import PerplexityEvaluation
from theseus.registry import evaluation


@dataclass
class Pes2OEvalConfig:
    num_samples: int = field("eval/pes2o_ppl/num_samples", default=500)


@evaluation("pes2o_ppl")
class Pes2OEval(PerplexityEvaluation):
    """Perplexity evaluation on peS2o scientific papers."""

    CONFIG = Pes2OEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset(
            "parquet",
            data_files=(
                "hf://datasets/allenai/peS2o@refs/convert/parquet/"
                "v2/partial-validation/*.parquet"
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
        return "pes2o_ppl"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]
