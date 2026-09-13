from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


def template(inputs: str, targets: str) -> ChatTemplate:
    return [
        ChatTurn(role="user", message=inputs),
        ChatTurn(role="assistant", message=targets),
    ]


@dataclass
class FlanConfig:
    split: str = field("data/flan/split", default="train")


@dataset("flan")
class Flan(ChatTemplateDataset):
    """Muennighoff/flan instruction-tuning dataset."""

    CONFIG = FlanConfig

    def __init__(self) -> None:
        split = configure(self.CONFIG).split
        # The HuggingFace dataset card records split sizes from an earlier
        # snapshot that don't match what's currently downloadable, so the
        # default 'basic_checks' verification raises NonMatchingSplitsSizesError.
        # Skip those checks — the actual data is fine for IFT.
        self.ds = load_dataset(
            "Muennighoff/flan", split=split, verification_mode="no_checks"
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        return template(item["inputs"], item["targets"])
