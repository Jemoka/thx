from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


def template(context: str, question: str, answer: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Read the following passage and answer the question. "
                "Respond with only the answer extracted from the passage.\n\n"
                f"Passage: {context}\n\n"
                f"Question: {question}"
            ),
        ),
        ChatTurn(role="assistant", message=answer),
    ]


@dataclass
class SQuADConfig:
    split: str = field("data/squad/split", default="train")


@dataset("squad")
class SQuAD(ChatTemplateDataset):
    """SQuAD v1.1 (Rajpurkar et al., 2016).

    Extractive question answering over Wikipedia paragraphs.
    """

    CONFIG = SQuADConfig

    def __init__(self) -> None:
        self.ds = load_dataset("rajpurkar/squad", split=configure(self.CONFIG).split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        answer = item["answers"]["text"][0] if item["answers"]["text"] else ""
        return template(item["context"], item["question"], answer)
