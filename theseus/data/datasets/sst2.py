from dataclasses import dataclass

from datasets import load_dataset
from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


def template(sentence: str, label: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Classify the sentiment of the following sentence; respond with "positive" or "negative", not including quotes.

sentence: {sentence}
""",
        ),
        ChatTurn(role="assistant", message=label),
    ]


@dataclass
class SST2Config:
    split: str = field("data/sst2/split", default="train")


@dataset("sst2")
class SST2(ChatTemplateDataset):
    CONFIG = SST2Config

    def __init__(self) -> None:
        self.ds = load_dataset("stanfordnlp/sst2", split=configure(self.CONFIG).split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        label = "positive" if item["label"] == 1 else "negative"
        return template(item["sentence"], label)
