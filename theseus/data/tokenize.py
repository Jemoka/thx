"""Node-backed tokenization jobs for statically declared datasets.

The materialized binary files are the state. Logged nodes only record offsets
that are safe after a flush, so these jobs use ``LoggingJob`` rather than the
model-oriented ``CheckpointedJob`` machinery.
"""

import itertools
import json
import random
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Mapping, TypeVar, cast

import numpy as np
from loguru import logger

from theseus.config import configure, field
from theseus.data.datasets import (
    ChatTemplate,
    ChatTemplateDataset,
    ContrastiveDataset,
    DatasetComponent,
    DatasetConfig,
    StreamingChatTemplateDataset,
    StreamingStringDataset,
    StringDataset,
)
from theseus.data.tokenizer import (
    TokenizerConfig,
    encode_chat_template,
    encode_chat_template_with_mask,
    get_tokenizer,
)
from theseus.job import LoggingJob
from theseus.store import ValueRow


#### helpers ####


def _encode_dataset_item(
    item: Any,
    is_chat: bool,
    tokenizer: Any,
    args: "TokenizeDatasetConfig | TokenizeContrastiveDatasetConfig",
) -> tuple[list[int], list[bool] | None]:
    """Encode a dataset item and its optional assistant mask."""
    if is_chat:
        chat_item = cast(ChatTemplate, item)
        if args.assistant_only:
            return encode_chat_template_with_mask(
                chat_item, tokenizer, args.system_prompt
            )
        return (
            encode_chat_template(
                chat_item, tokenizer, args.system_prompt, tokenize=True
            ),
            None,
        )
    return tokenizer.encode(cast(str, item)), None


def _build_padded_arrays(
    ids: list[int],
    token_mask: list[bool] | None,
    block_size: int,
    pad_token: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncate or left-pad token ids and their loss mask."""
    ids = ids[-block_size:]
    token_mask = token_mask[-block_size:] if token_mask is not None else None
    padding = block_size - len(ids)
    mask = (
        [False] * padding + token_mask
        if token_mask is not None
        else [False] * padding + [True] * len(ids)
    )
    return (
        np.asarray([pad_token] * padding + ids, dtype=np.uint32),
        np.asarray(mask, dtype=np.bool_),
    )


def _open_memmap(path: Path, dtype: Any, shape: tuple[int, ...]) -> np.memmap[Any, Any]:
    """Open an existing materialization for update or create it."""
    mode: Literal["r+", "w+"] = "r+" if path.exists() else "w+"
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


#### configuration ####


@dataclass
class TokenizeConfig:
    """Configuration shared by every tokenization strategy."""

    val_pct: float = field("data/val_pct", default=0.05)
    seed: int = field("data/seed", default=2357)


@dataclass
class TokenizeDatasetConfig(TokenizeConfig):
    """Fixed-width tokenization; replacement=False deduplicates source rows."""

    block_size: int = field("architecture/block_size", default=512)
    pad_token: int = field("data/pad_token", default=0)
    system_prompt: str = field("data/system_prompt", default="")
    assistant_only: bool = field("data/assistant_only", default=False)
    replacement: bool = field("data/tokenize/replacement", default=True)
    allow_truncation: bool = field("data/allow_truncation", default=True)


@dataclass
class TokenizePretrainingDatasetConfig(TokenizeConfig):
    """Configuration for variable-width streaming tokenization."""

    max_samples: int = field("data/max_samples", default=-1)
    max_tokens: int = field("data/max_tokens", default=-1)


@dataclass
class TokenizeContrastiveDatasetConfig(TokenizeDatasetConfig):
    """Configuration for paired fixed-width tokenization."""


C = TypeVar("C", bound=TokenizeConfig)


#### job lifecycle ####


class TokenizeJob(LoggingJob[C], Generic[C]):
    """Materialize one statically declared raw dataset.

    Progress is durable only after the output files are flushed and the
    corresponding node metadata is written. Both resume and branch continue
    from that durable point; their distinction is only the resulting DAG edge.
    These jobs deliberately do not infer progress from files or legacy JSON.
    """

    DATASET: type[DatasetComponent]
    CONFIG: type[C]

    @classmethod
    def config(cls) -> list[type[Any]]:
        """Return tokenizer, dataset, and strategy configuration schemas.

        Returns:
            Schemas needed to construct this concrete tokenization job.
        """
        return [cls.CONFIG, TokenizerConfig, *cls.DATASET.config()]

    @property
    def output_path(self) -> Path:
        """Return the legacy-compatible materialized dataset path.

        Returns:
            ``<data>/<dataset-key>`` or ``<data>/<dataset-key>_<suffix>``.
        """
        data_dir = self.spec.hardware.hosts[0].cluster.data_dir
        suffix = configure(DatasetConfig).suffix
        name = self.DATASET.DATASET_KEY
        return data_dir / (name if suffix == "" else f"{name}_{suffix}")

    def state_init(self) -> None:
        """Initialize fresh progress and reject an existing output path."""
        self.progress: dict[str, bool | float | int | str] = {
            "data/dataset": self.DATASET.DATASET_KEY,
            "data/suffix": configure(DatasetConfig).suffix,
            "data/complete": False,
        }
        if self.main_process() and self.output_path.exists():
            raise FileExistsError(
                f"Dataset output already exists at {self.output_path}; "
                "restore a node or choose another data/suffix."
            )

    def state_restore(self, state: ValueRow) -> None:
        """Restore progress from a node and validate its materialization."""
        suffix = configure(DatasetConfig).suffix
        if state.get("data/dataset") != self.DATASET.DATASET_KEY:
            raise ValueError("The restored node belongs to another dataset.")
        if state.get("data/suffix", "") != suffix:
            raise ValueError("The restored node uses another dataset suffix.")
        if self.main_process() and not self.output_path.exists():
            raise FileNotFoundError(self.output_path)
        self.progress = {
            key: value
            for key, value in state.items()
            if key.startswith("data/") and isinstance(value, (bool, float, int, str))
        }
        if self.main_process():
            self._validate_progress()

    @abstractmethod
    def _validate_progress(self) -> None:
        """Validate files required by the restored counters."""

    def restored_complete(self) -> bool:
        """Advance resumed tokenization and re-publish completion when present.

        Returns:
            Whether the restored materialization was already complete.
        """
        if self.base is not None and self.node.serialize() == self.base.serialize():
            self.tick()
        if not bool(self.progress.get("data/complete", False)):
            return False
        self.log(self.progress)
        return True

    def publish(self, progress: Mapping[str, bool | float | int | str]) -> None:
        """Publish a durable progress point and advance the node.

        Args:
            progress: Counters describing files already flushed to disk.
        """
        self.progress.update(progress)
        self.log(self.progress)
        self.tick()


#### fixed-width tokenization ####


class TokenizeBlockwiseDatasetJob(TokenizeJob[TokenizeDatasetConfig]):
    """Materialize one indexable dataset as padded token and mask arrays."""

    CONFIG = TokenizeDatasetConfig

    def _validate_progress(self) -> None:
        split = str(self.progress.get("data/split", "train"))
        rows = int(self.progress.get("data/split_index", 0))
        for suffix, size in (("bin", 4), ("bin.mask", 1)):
            path = self.output_path / f"{split}.{suffix}"
            if (
                not path.exists()
                or path.stat().st_size < rows * self.args.block_size * size
            ):
                raise ValueError(f"Dataset file is behind restored state: {path}")
        if (
            bool(self.progress.get("data/complete", False))
            and not (self.output_path / "shape.json").exists()
        ):
            raise ValueError("Completed tokenization is missing shape.json.")

    def run(self) -> None:
        """Tokenize the configured dataset, continuing from restored rows."""
        if not self.main_process() or self.restored_complete():
            return
        args = self.args
        output_path = self.output_path
        output_path.mkdir(parents=True, exist_ok=True)
        dataset = cast(
            StringDataset | ChatTemplateDataset,
            self.DATASET(),
        )
        tokenizer = get_tokenizer()
        is_chat = isinstance(dataset[0], list)
        indices = list(range(len(dataset)))
        if not args.replacement:
            seen: set[Any] = set()
            indices = []
            for index in range(len(dataset)):
                item = dataset[index]
                key = tuple(item) if is_chat else item
                if key not in seen:
                    seen.add(key)
                    indices.append(index)
        random.Random(args.seed).shuffle(indices)
        val_size = int(len(indices) * args.val_pct)
        splits = {"train": indices[val_size:], "val": indices[:val_size]}
        restored_split = str(self.progress.get("data/split", "train"))
        restored_index = int(self.progress.get("data/split_index", 0))
        passed_restore = "data/split" not in self.progress
        completed = 0

        for split_name, split_indices in splits.items():
            if not passed_restore and split_name != restored_split:
                completed += len(split_indices)
                continue
            start = restored_index if split_name == restored_split else 0
            passed_restore = True
            count = len(split_indices)
            token_path = output_path / f"{split_name}.bin"
            mask_path = output_path / f"{split_name}.bin.mask"
            tokens_array = _open_memmap(token_path, np.uint32, (count, args.block_size))
            mask_array = _open_memmap(mask_path, np.bool_, (count, args.block_size))
            interval = max(1, count // 20)
            started = time.time()

            for row, dataset_index in enumerate(split_indices[start:], start=start):
                ids, token_mask = _encode_dataset_item(
                    dataset[dataset_index], is_chat, tokenizer, args
                )
                if not args.allow_truncation and len(ids) > args.block_size:
                    raise ValueError(
                        f"Example {dataset_index} exceeds block_size; truncation is disabled"
                    )
                tokens_array[row], mask_array[row] = _build_padded_arrays(
                    ids, token_mask, args.block_size, args.pad_token
                )
                if (row + 1) % interval == 0:
                    tokens_array.flush()
                    mask_array.flush()
                    self.publish(
                        {
                            "data/split": split_name,
                            "data/split_index": row + 1,
                            "data/samples": completed + row + 1,
                            "data/rate": (row + 1 - start)
                            / max(time.time() - started, 1e-9),
                        }
                    )

            tokens_array.flush()
            mask_array.flush()
            if count % interval != 0:
                self.publish(
                    {
                        "data/split": split_name,
                        "data/split_index": count,
                        "data/samples": completed + count,
                    }
                )
            completed += count

        with open(output_path / "shape.json", "w") as file:
            json.dump(
                {name: [len(rows), args.block_size] for name, rows in splits.items()},
                file,
                indent=4,
            )
        self.publish({"data/complete": True})


#### contrastive tokenization ####


class TokenizeContrastiveDatasetJob(TokenizeJob[TokenizeContrastiveDatasetConfig]):
    """Materialize one paired dataset as padded positive/negative arrays."""

    CONFIG = TokenizeContrastiveDatasetConfig

    def _validate_progress(self) -> None:
        split = str(self.progress.get("data/split", "train"))
        rows = int(self.progress.get("data/split_index", 0))
        for side in ("pos", "neg"):
            for suffix, size in (("bin", 4), ("bin.mask", 1)):
                path = self.output_path / f"{split}.{side}.{suffix}"
                if (
                    not path.exists()
                    or path.stat().st_size < rows * self.args.block_size * size
                ):
                    raise ValueError(f"Dataset file is behind restored state: {path}")
        if (
            bool(self.progress.get("data/complete", False))
            and not (self.output_path / "shape.json").exists()
        ):
            raise ValueError("Completed tokenization is missing shape.json.")

    def run(self) -> None:
        """Tokenize paired examples, continuing from restored rows."""
        if not self.main_process() or self.restored_complete():
            return
        args = self.args
        output_path = self.output_path
        output_path.mkdir(parents=True, exist_ok=True)
        dataset = cast(
            ContrastiveDataset[Any],
            self.DATASET(),
        )
        tokenizer = get_tokenizer()
        is_chat = isinstance(dataset[0][0], list)
        indices = list(range(len(dataset)))
        if not args.replacement:
            seen: set[Any] = set()
            indices = []
            for index in range(len(dataset)):
                item = dataset[index]
                key = tuple(tuple(side) if is_chat else side for side in item)
                if key not in seen:
                    seen.add(key)
                    indices.append(index)
        random.Random(args.seed).shuffle(indices)
        val_size = int(len(indices) * args.val_pct)
        splits = {"train": indices[val_size:], "val": indices[:val_size]}
        restored_split = str(self.progress.get("data/split", "train"))
        restored_index = int(self.progress.get("data/split_index", 0))
        passed_restore = "data/split" not in self.progress
        completed = 0

        for split_name, split_indices in splits.items():
            if not passed_restore and split_name != restored_split:
                completed += len(split_indices)
                continue
            start = restored_index if split_name == restored_split else 0
            passed_restore = True
            count = len(split_indices)
            paths = {
                side: (
                    output_path / f"{split_name}.{side}.bin",
                    output_path / f"{split_name}.{side}.bin.mask",
                )
                for side in ("pos", "neg")
            }
            arrays = {
                side: (
                    _open_memmap(token_path, np.uint32, (count, args.block_size)),
                    _open_memmap(mask_path, np.bool_, (count, args.block_size)),
                )
                for side, (token_path, mask_path) in paths.items()
            }
            interval = max(1, count // 20)

            for row, dataset_index in enumerate(split_indices[start:], start=start):
                for side, item in zip(("pos", "neg"), dataset[dataset_index]):
                    ids, token_mask = _encode_dataset_item(
                        item, is_chat, tokenizer, args
                    )
                    if not args.allow_truncation and len(ids) > args.block_size:
                        raise ValueError("Dataset example exceeds data/block_size")
                    arrays[side][0][row], arrays[side][1][row] = _build_padded_arrays(
                        ids, token_mask, args.block_size, args.pad_token
                    )
                if (row + 1) % interval == 0:
                    for pair in arrays.values():
                        pair[0].flush()
                        pair[1].flush()
                    self.publish(
                        {
                            "data/split": split_name,
                            "data/split_index": row + 1,
                            "data/samples": completed + row + 1,
                        }
                    )

            for pair in arrays.values():
                pair[0].flush()
                pair[1].flush()
            if count % interval != 0:
                self.publish(
                    {
                        "data/split": split_name,
                        "data/split_index": count,
                        "data/samples": completed + count,
                    }
                )
            completed += count

        with open(output_path / "shape.json", "w") as file:
            json.dump(
                {
                    name: {
                        "pos": [len(rows), args.block_size],
                        "neg": [len(rows), args.block_size],
                    }
                    for name, rows in splits.items()
                },
                file,
                indent=4,
            )
        self.publish({"data/complete": True})


#### variable-width tokenization ####


class TokenizeVariableDatasetJob(TokenizeJob[TokenizePretrainingDatasetConfig]):
    """Materialize one streaming dataset as contiguous train/validation tokens."""

    CONFIG = TokenizePretrainingDatasetConfig

    def _validate_progress(self) -> None:
        for name in ("train", "val"):
            offset = int(self.progress.get(f"data/{name}_tokens", 0))
            path = self.output_path / f"{name}.bin"
            if not path.exists() or path.stat().st_size < offset * 4:
                raise ValueError(f"Token file is behind restored state: {path}")

    def run(self) -> None:
        """Tokenize a stream, continuing from restored token offsets."""
        if not self.main_process() or self.restored_complete():
            return
        args = self.args
        output_path = self.output_path
        output_path.mkdir(parents=True, exist_ok=True)
        dataset = cast(
            StreamingStringDataset | StreamingChatTemplateDataset,
            self.DATASET(),
        )
        iterator = iter(dataset)
        first = next(iterator)
        items = itertools.chain((first,), iterator)
        is_chat = isinstance(first, list)
        tokenizer = get_tokenizer()
        train_path = output_path / "train.bin"
        val_path = output_path / "val.bin"
        train_index = int(self.progress.get("data/train_tokens", 0))
        val_index = int(self.progress.get("data/val_tokens", 0))
        sample_count = int(self.progress.get("data/samples", 0))
        resume_samples = sample_count
        restored = "data/samples" in self.progress

        for path, offset in ((train_path, train_index), (val_path, val_index)):
            if restored:
                with open(path, "r+b") as file:
                    file.truncate(offset * 4)

        train_size = max(1_000_000, train_index + 1_000_000)
        val_size = max(1_000_000, val_index + 1_000_000)
        for path, size in ((train_path, train_size), (val_path, val_size)):
            if path.exists():
                with open(path, "r+b") as file:
                    file.truncate(size * 4)
        train_array = _open_memmap(train_path, np.uint32, (train_size,))
        val_array = _open_memmap(val_path, np.uint32, (val_size,))
        rng = random.Random(args.seed)
        started = time.time()
        previous_time = started
        previous_samples = sample_count
        previous_tokens = train_index + val_index
        visited = 0

        for seen, item in enumerate(items):
            visited = seen + 1
            if seen < sample_count:
                rng.random()
                continue
            if args.max_samples >= 0 and seen >= args.max_samples:
                break
            if args.max_tokens >= 0 and train_index + val_index >= args.max_tokens:
                break
            sample_count = seen + 1
            if is_chat:
                ids = encode_chat_template(
                    cast(ChatTemplate, item), tokenizer, tokenize=True
                )
            else:
                ids = tokenizer.encode_ordinary(cast(str, item))
                ids.append(tokenizer.eot_token)
            sample = np.asarray(ids, dtype=np.uint32)
            if rng.random() < args.val_pct:
                if val_index + len(sample) > len(val_array):
                    val_array.flush()
                    val_array._mmap.close()  # type: ignore[attr-defined]
                    val_size = max(len(val_array) * 2, val_index + len(sample))
                    with open(val_path, "r+b") as file:
                        file.truncate(val_size * 4)
                    val_array = np.memmap(
                        val_path, dtype=np.uint32, mode="r+", shape=(val_size,)
                    )
                val_array[val_index : val_index + len(sample)] = sample
                val_index += len(sample)
            else:
                if train_index + len(sample) > len(train_array):
                    train_array.flush()
                    train_array._mmap.close()  # type: ignore[attr-defined]
                    train_size = max(len(train_array) * 2, train_index + len(sample))
                    with open(train_path, "r+b") as file:
                        file.truncate(train_size * 4)
                    train_array = np.memmap(
                        train_path, dtype=np.uint32, mode="r+", shape=(train_size,)
                    )
                train_array[train_index : train_index + len(sample)] = sample
                train_index += len(sample)

            if sample_count % 1000 == 0:
                train_array.flush()
                val_array.flush()
                now = time.time()
                self.publish(
                    {
                        "data/samples": sample_count,
                        "data/train_tokens": train_index,
                        "data/val_tokens": val_index,
                        "data/sample_rate": (sample_count - previous_samples)
                        / max(now - previous_time, 1e-9),
                        "data/token_rate": (train_index + val_index - previous_tokens)
                        / max(now - previous_time, 1e-9),
                    }
                )
                previous_time = now
                previous_samples = sample_count
                previous_tokens = train_index + val_index
                logger.info(
                    "DATA | {} samples | {} train tokens | {} val tokens | {:.1f}s",
                    sample_count,
                    train_index,
                    val_index,
                    now - started,
                )

        if visited < resume_samples:
            train_array._mmap.close()  # type: ignore[attr-defined]
            val_array._mmap.close()  # type: ignore[attr-defined]
            raise ValueError("Dataset stream ended before the restored sample offset.")

        train_array.flush()
        train_array._mmap.close()  # type: ignore[attr-defined]
        val_array.flush()
        val_array._mmap.close()  # type: ignore[attr-defined]
        with open(train_path, "r+b") as file:
            file.truncate(train_index * 4)
        with open(val_path, "r+b") as file:
            file.truncate(val_index * 4)
        self.publish(
            {
                "data/samples": sample_count,
                "data/train_tokens": train_index,
                "data/val_tokens": val_index,
                "data/complete": True,
            }
        )


__all__ = [
    "TokenizeBlockwiseDatasetJob",
    "TokenizeContrastiveDatasetJob",
    "TokenizeDatasetConfig",
    "TokenizeJob",
    "TokenizePretrainingDatasetConfig",
    "TokenizeVariableDatasetJob",
]
