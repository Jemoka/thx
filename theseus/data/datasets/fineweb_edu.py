from pathlib import Path

from theseus.config import configure
from theseus.data.datasets.dataset import DatasetComponent, DatasetConfig
from theseus.registry import dataset


@dataset("track_m_fineweb_edu_dedup")
class FineWebEduDedup(DatasetComponent):
    """Deduplicated FineWeb-Edu from the Track-M preprocessing pipeline.

    This dataset is consumed as pre-tokenized memmaps placed under
    ``$root/data/track_m_fineweb_edu_dedup`` (or ``..._<suffix>`` when
    ``data/suffix`` is set), with a required ``train.bin`` and optional
    ``val.bin``. Both files must be contiguous ``np.uint32`` token-id streams
    (``cl100k_base`` tokenization), matching the PMD loader format.
    """

    def __init__(self) -> None:
        super().__init__()
        suffix = configure(DatasetConfig).suffix
        dataset_dir = (
            self.DATASET_KEY if suffix == "" else f"{self.DATASET_KEY}_{suffix}"
        )
        train_path = Path("data") / dataset_dir / "train.bin"
        if not train_path.exists():
            raise FileNotFoundError(
                "FineWeb-Edu memmap is missing. "
                f"Expected {train_path} in the current $root/data layout. "
                "Place the Track-M export at "
                f"$root/data/{dataset_dir}/train.bin "
                "(and optional val.bin) as contiguous uint32 token ids."
            )
