from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


def template(question: str, query: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Translate the following natural language question into a "
                "SPARQL query. Respond with only the SPARQL query, nothing else.\n\n"
                f"Question: {question}"
            ),
        ),
        ChatTurn(role="assistant", message=query),
    ]


@dataclass
class CFQConfig:
    subset: str = field("data/cfq/subset", default="mcd1")
    split: str = field("data/cfq/split", default="train")


@dataset("cfq")
class CFQ(ChatTemplateDataset):
    """Compositional Freebase Questions (Google).

    Each example maps a natural language question to a SPARQL query.
    The ``config`` parameter selects the MCD split (default ``"mcd1"``).
    """

    CONFIG = CFQConfig

    def __init__(self) -> None:
        args = configure(self.CONFIG)
        self.ds = load_dataset(
            "google-research-datasets/cfq", args.subset, split=args.split
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        return template(item["question"], item["query"])
