import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

from theseus.config import configure, field
from theseus.data.datasets import StreamingPretrainingDataset
from theseus.registry import dataset

_REPO_ID = "allenai/peS2o"
_REPO_TYPE = "dataset"

_SHARDS = {
    "v1": {
        "train": [f"data/v1/train-{i:05d}-of-00020.json.gz" for i in range(20)],
        "validation": [
            f"data/v1/validation-{i:05d}-of-00002.json.gz" for i in range(2)
        ],
    },
    "v2": {
        "train": [f"data/v2/train-{i:05d}-of-00020.json.gz" for i in range(20)],
        "validation": [
            f"data/v2/validation-{i:05d}-of-00002.json.gz" for i in range(2)
        ],
    },
}


@dataclass
class Pes2OConfig:
    version: str = field("data/pes2o/version", default="v2")
    split: str = field("data/pes2o/split", default="train")


@dataset("pes2o")
class Pes2O(StreamingPretrainingDataset):
    """Scientific papers from the peS2o corpus (AllenAI).

    Streams gzipped JSONL shards directly from the HF repo, bypassing the
    deprecated loading script (removed in ``datasets`` 4.0). Each shard is
    downloaded via ``hf_hub_download`` (cached locally) then read line by line.
    """

    CONFIG = Pes2OConfig

    def __init__(self) -> None:
        args = configure(self.CONFIG)
        split = "validation" if args.split == "val" else args.split
        self._shards = _SHARDS[args.version][split]

    def __iter__(self) -> Iterator[str]:
        for shard in self._shards:
            local_path = hf_hub_download(
                repo_id=_REPO_ID,
                filename=shard,
                repo_type=_REPO_TYPE,
            )
            with gzip.open(local_path, "rt", encoding="utf-8") as f:
                for line in f:
                    text = json.loads(line).get("text")
                    if text:
                        yield text
