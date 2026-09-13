from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset
from theseus.config import configure, field
from theseus.data.datasets import StreamingPretrainingDataset
from theseus.registry import dataset


@dataclass
class FineWebConfig:
    snapshot: str = field("data/fineweb/snapshot", default="CC-MAIN-2022-21")


@dataset("fineweb")
class FineWeb(StreamingPretrainingDataset):
    CONFIG = FineWebConfig

    def __init__(self) -> None:
        self.ds = load_dataset(
            "HuggingFaceFW/fineweb",
            configure(self.CONFIG).snapshot,
            split="train",
            streaming=True,
        )

    def __iter__(self) -> Iterator[str]:
        for i in self.ds["text"]:
            yield i
