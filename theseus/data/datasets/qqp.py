from dataclasses import dataclass

from datasets import load_dataset
from theseus.config import configure, field
from theseus.data.datasets import ChatTemplateDataset, ChatTemplate, ChatTurn
from theseus.registry import dataset


def template(q1: str, q2: str, label: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=f"""Are these two questions paraphrases of each other? Answer with "yes" or "no", not including quotes.

question 1: {q1}
question 2: {q2}
""",
        ),
        ChatTurn(role="assistant", message=label),
    ]


@dataclass
class QQPConfig:
    split: str = field("data/qqp/split", default="train")


@dataset("qqp")
class QQP(ChatTemplateDataset):
    CONFIG = QQPConfig

    def __init__(self) -> None:
        self.ds = load_dataset(
            "nyu-mll/glue", "qqp", split=configure(self.CONFIG).split
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        return template(
            item["question1"], item["question2"], "yes" if item["label"] else "no"
        )
