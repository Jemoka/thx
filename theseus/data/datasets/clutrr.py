from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


def template(story: str, query: str, target_text: str) -> ChatTemplate:
    return [
        ChatTurn(
            role="user",
            message=(
                "Read the following story about a family and determine the "
                "relationship between the two people mentioned in the query. "
                "Respond with only the relationship (e.g. aunt, grandfather, "
                "brother, sister, father, mother, etc.).\n\n"
                f"Story: {story}\n\n"
                f"Query: {query}"
            ),
        ),
        ChatTurn(role="assistant", message=target_text),
    ]


@dataclass
class CLUTRRConfig:
    subset: str = field("data/clutrr/subset", default="gen_train234_test2to10")
    split: str = field("data/clutrr/split", default="train")


@dataset("clutrr")
class CLUTRR(ChatTemplateDataset):
    """CLUTRR relational reasoning benchmark (Facebook Research).

    Given a semi-synthetic story about a hypothetical family, infer the
    kinship relation between two specified family members.  The ``config``
    parameter selects the subset (default ``"gen_train234_test2to10"``).
    """

    CONFIG = CLUTRRConfig

    def __init__(self) -> None:
        args = configure(self.CONFIG)
        self.ds = load_dataset(
            "parquet",
            data_files=(
                f"hf://datasets/CLUTRR/v1@refs/convert/parquet/"
                f"{args.subset}/{args.split}/*.parquet"
            ),
            split="train",
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        return template(item["story"], item["query"], item["target_text"])
