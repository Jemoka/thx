from collections.abc import Iterator

from datasets import load_dataset

from theseus.data.datasets import StreamingPretrainingDataset
from theseus.registry import dataset


@dataset("pile_detoxify")
class PileDetoxify(StreamingPretrainingDataset):
    """Filtered Pile with toxicity scores (Korbak et al.).

    Streams text from ``tomekkorbak/pile-detoxify``, which annotates
    Pile documents with per-sentence toxicity scores from Detoxify.
    Each yielded string is the full document text (sentences joined).
    """

    def __init__(self) -> None:
        self.ds = load_dataset(
            "tomekkorbak/pile-detoxify",
            split="train",
            streaming=True,
        )

    def __iter__(self) -> Iterator[str]:
        for item in self.ds:
            texts = item.get("texts", [])
            if texts:
                yield " ".join(texts)
