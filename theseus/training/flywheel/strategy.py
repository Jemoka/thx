"""Data loading for statically declared dataset mixtures."""

from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

from enum import Enum

from theseus.base.job import ExecutionSpec
from theseus.config import configure
from theseus.data.datasets import DatasetComponent, DatasetConfig
from .dataset import Dataset
from .stream import batches
from theseus.base import Node
import copy
import numpy as np
from collections.abc import Iterator
from typing import Dict
from .stream import AsyncStrategy as AsyncStrategy


class DatasetStyle(Enum):
    PADDED = "padded"
    PMD = "pmd"
    CONTRASTIVE = "contrastive"

    @classmethod
    def normalize(cls, style: "DatasetStyleLike") -> "DatasetStyle":
        if isinstance(style, cls):
            return style
        style_lower = str(style).lower()
        for option in cls:
            if style_lower == option.value:
                return option
        raise ValueError(f"Unknown dataset style: {style}")


DatasetStyleLike = DatasetStyle | Literal["padded", "pmd", "contrastive"]


@dataclass
class Sampling:
    """Weight and reader style for one statically declared dataset."""

    dataset: type[DatasetComponent]
    rate: float
    if TYPE_CHECKING:
        style: DatasetStyleLike = DatasetStyle.PADDED
    else:
        style: Any = DatasetStyle.PADDED


class Strategy:
    @staticmethod
    def config(mixture: list[Sampling]) -> list[type[Any]]:
        """Return configuration schemas required by a dataset mixture.

        Args:
            mixture: Static dataset mixture declared by a trainer.

        Returns:
            Dataset configuration schemas in declaration order.
        """

        return [schema for sample in mixture for schema in sample.dataset.config()]

    def __init__(self, spec: ExecutionSpec, block_size: int, mixture: list[Sampling]):
        self.spec = spec
        self.block_size = block_size

        # validate that rates sum to 1
        self.rates = [sampling.rate for sampling in mixture]
        total_rate = sum(self.rates)
        if any(rate < 0 for rate in self.rates) or not abs(total_rate - 1.0) < 1e-6:
            raise ValueError(f"Sampling rates must sum to 1, got {total_rate}")
        self.rates = [rate / total_rate for rate in self.rates]

        # Create dataset objects
        self.datasets: list[Dataset] = []
        styles_lower: list[str] = []
        suffix = configure(DatasetConfig).suffix
        for sampling in mixture:
            ds: Dataset
            style = DatasetStyle.normalize(sampling.style)
            styles_lower.append(style.value)

            if style == DatasetStyle.PADDED:
                from theseus.training.flywheel.padded import PaddedDataset

                ds = PaddedDataset(
                    spec, block_size, sampling.dataset.DATASET_KEY, suffix
                )
            elif style == DatasetStyle.CONTRASTIVE:
                from theseus.training.flywheel.contrastive import (
                    ContrastivePaddedDataset,
                )

                ds = ContrastivePaddedDataset(
                    spec, block_size, sampling.dataset.DATASET_KEY, suffix
                )
            elif style == DatasetStyle.PMD:
                from theseus.training.flywheel.pmd import MemmapDataset

                ds = MemmapDataset(
                    spec, block_size, sampling.dataset.DATASET_KEY, suffix
                )
            self.datasets.append(ds)

        if DatasetStyle.CONTRASTIVE.value in styles_lower and any(
            style != DatasetStyle.CONTRASTIVE.value for style in styles_lower
        ):
            raise ValueError("Contrastive and causal datasets cannot share a mixture")

    def get_async_batches(
        self,
        batch_size: int,
        split: str = "train",
        node: Node | None = None,
        seed: int = 0,
    ) -> AsyncStrategy:
        # Readers contain memmap/buffer caches; each worker owns its readers.
        datasets = [copy.copy(dataset) for dataset in self.datasets]
        for dataset in datasets:
            if hasattr(dataset, "_reader"):
                dataset._reader = copy.copy(dataset._reader)
            if hasattr(dataset, "_buffers"):
                dataset._buffers = {}
        return AsyncStrategy(datasets, self.rates, batch_size, split, node, seed)

    def get_batch(self, batch_size: int, split: str = "train") -> Dict[str, np.ndarray]:
        if not hasattr(self, "_streams"):
            self._streams: dict[tuple[int, str], Iterator[Dict[str, np.ndarray]]] = {}
        key = (batch_size, split)
        if key not in self._streams:
            self._streams[key] = batches(self.datasets, self.rates, batch_size, split)
        return next(self._streams[key])
