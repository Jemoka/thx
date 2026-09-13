from dataclasses import dataclass

from datasets import load_dataset
from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


def template(sentence: str, option1: str, option2: str, label: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Fill in the blank (represented by "_") in the sentence. Answer in a single letter, "A" or "B", without quotes.

sentence: {sentence}

A: {option1}
B: {option2}
""",
        ),
        ChatTurn(role="assistant", message=label),
    ]


@dataclass
class WinograndeConfig:
    split: str = field("data/winogrande/split", default="train")


@dataset("winogrande")
class Winogrande(ChatTemplateDataset):
    CONFIG = WinograndeConfig

    def __init__(self) -> None:
        self.ds = load_dataset(
            "allenai/winogrande",
            "winogrande_xl",
            split=configure(self.CONFIG).split,
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        label = "A" if item["answer"] == "1" else "B"
        return template(item["sentence"], item["option1"], item["option2"], label)
