from dataclasses import dataclass
from datasets import load_dataset
from typing import Tuple

from theseus.config import configure, field
from theseus.evaluation.base import PerplexityComparisonEvaluation
from theseus.registry import evaluation


@dataclass
class LongBenchConfig:
    split: str = field("eval/longbench/split", default="train")


@evaluation("longbench")
class LongBench(PerplexityComparisonEvaluation):
    CONFIG = LongBenchConfig

    def __init__(self) -> None:
        self.ds = load_dataset("THUDM/LongBench-v2", split=configure(self.CONFIG).split)

    @property
    def name(self) -> str:
        return "longbench"

    def get(self, indx: int) -> Tuple[str, list[str], int]:
        item = self.ds[indx]
        continuations = [
            item["choice_A"],
            item["choice_B"],
            item["choice_C"],
            item["choice_D"],
        ]
        correct_idx = ord(item["answer"].strip().upper()) - ord("A")
        prefix = (
            f"Context:\n{item['context']}\n\nQuestion: {item['question']}\n\nAnswer: "
        )
        return prefix, continuations, correct_idx

    def __len__(self) -> int:
        return len(self.ds)
