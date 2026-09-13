from collections.abc import Iterator

from datasets import load_dataset

from theseus.data.datasets import StreamingPretrainingDataset
from theseus.registry import dataset


@dataset("pile")
class Pile(StreamingPretrainingDataset):
    """The Pile (EleutherAI) for general pretraining.

    Streams text from ``EleutherAI/pile``, an 825 GiB diverse
    open-source language modelling dataset.  Uses parquet auto-convert
    to bypass deprecated custom loading scripts.
    """

    def __init__(self) -> None:
        self.ds = load_dataset(
            "parquet",
            data_files=(
                "hf://datasets/EleutherAI/pile@refs/convert/parquet/"
                "all/partial-train/*.parquet"
            ),
            split="train",
            streaming=True,
        )

    def __iter__(self) -> Iterator[str]:
        for item in self.ds:
            text = item.get("text", "")
            if text:
                yield text
